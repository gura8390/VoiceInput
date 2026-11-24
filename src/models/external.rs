//! Byte-preserving ONNX external-data conversion.
//! Nodes, attributes, declarations, dtypes, dimensions and weight payloads stay intact.
//! This path never creates an ONNX Runtime session.
use super::{hex, sha256_file};
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{Read, Seek, SeekFrom, Write},
    path::{Component, Path, PathBuf},
};

const CHUNK: u64 = 1024 * 1024;

#[derive(Clone, Debug)]
pub(crate) struct PrepareOptions {
    pub minimum: u64,
    pub alignment: u64,
    pub expected_source_sha256: Option<String>,
}

impl Default for PrepareOptions {
    fn default() -> Self {
        Self {
            minimum: 4096,
            alignment: 4096,
            expected_source_sha256: None,
        }
    }
}

#[derive(Clone, Copy)]
struct Field {
    number: u64,
    wire: u8,
    start: u64,
    payload: u64,
    end: u64,
    scalar: Option<u64>,
}

struct TensorView {
    name: String,
    dtype: u64,
    dimensions: Vec<u64>,
    raw_bytes: u64,
    data_location: u64,
    external: BTreeMap<String, String>,
    field: Field,
    inner: Vec<Field>,
    raw_field: Option<Field>,
    #[allow(dead_code)]
    metadata_sha256: String,
}

struct Inspection {
    bytes: u64,
    graph_fields: Vec<Field>,
    top: Vec<Field>,
    tensors: Vec<TensorView>,
    initializer_metadata_sha256: String,
    other_graph_fields_sha256: String,
    other_model_fields_sha256: String,
}

pub(crate) fn prepare(
    source: &Path,
    output: &Path,
    output_root: &Path,
    options: &PrepareOptions,
) -> Result<Value> {
    let (source, output, output_root, weights_path, receipt_path) =
        output_paths(source, output, output_root)?;
    ensure!(
        options.minimum >= 1
            && (1..=65536).contains(&options.alignment)
            && options.alignment.is_power_of_two(),
        "阈值或对齐无效"
    );
    let expected = match options.expected_source_sha256.as_deref() {
        Some(value) => Some(super::parse_sha256(value)?),
        None => None,
    };
    let before = file_stamp(&fs::symlink_metadata(&source)?)?;
    let mut src = File::open(&source)?;
    let original = inspect_file(&mut src, before.2)?;
    let source_sha = sha256_file(&source)?;
    if let Some(expected) = expected.as_deref()
        && source_sha != expected
    {
        bail!("源模型哈希与已接受的模型不一致");
    }
    let weights_name = weights_path
        .file_name()
        .context("权重文件名")?
        .to_string_lossy()
        .into_owned();
    let mut planned = Vec::new();
    let mut total = 0u64;
    for tensor in &original.tensors {
        if !tensor.external.is_empty() || tensor.data_location != 0 {
            bail!("源模型只能包含内嵌权重");
        }
        if tensor.raw_bytes < options.minimum {
            continue;
        }
        total = align_up(total, options.alignment)?;
        let extra = external_fields(&weights_name, total, tensor.raw_bytes);
        let remaining = tensor
            .inner
            .iter()
            .copied()
            .filter(|field| field.number != 9 && field.number != 14)
            .collect::<Vec<_>>();
        let body = remaining
            .iter()
            .try_fold(0u64, |sum, field| sum.checked_add(field.end - field.start))
            .context("initializer 长度溢出")?;
        let new_length = body
            .checked_add(extra.len() as u64)
            .context("initializer 长度溢出")?;
        planned.push((
            tensor.field.start,
            total,
            remaining,
            extra,
            new_length,
            tensor,
        ));
        total = total
            .checked_add(tensor.raw_bytes)
            .context("权重文件长度溢出")?;
    }
    if planned.is_empty() {
        bail!("没有可外置的 raw_data 权重");
    }
    let parent = output.parent().context("输出目录")?;
    fs::create_dir_all(parent)?;
    assert_no_reparse(parent)?;
    let mut identities = Vec::new();
    {
        let mut data = guarded_new_file(&weights_path, &output_root, false)?;
        for (_, offset, _, _, _, tensor) in &planned {
            let position = data.stream_position()?;
            ensure!(*offset >= position, "权重偏移回退");
            write_zeros(&mut data, *offset - position)?;
            let raw = tensor.raw_field.context("缺少 raw_data")?;
            let mut digest = Sha256::new();
            copy_span(
                &mut src,
                raw.payload,
                raw.end,
                &mut HashWriter {
                    inner: &mut data,
                    hasher: &mut digest,
                },
            )?;
            identities.push(json!({
                "name": tensor.name,
                "dtype": tensor.dtype,
                "dimensions": tensor.dimensions,
                "offset": offset,
                "bytes": tensor.raw_bytes,
                "payload_sha256": hex(&digest.finalize()),
            }));
        }
    }
    let graph_length = original.graph_fields.iter().try_fold(0u64, |sum, field| {
        let size = if let Some((_, _, _, _, new_length, _)) =
            planned.iter().find(|(start, ..)| *start == field.start)
        {
            (varint(5 * 8 + 2).len() as u64)
                .checked_add(varint(*new_length).len() as u64)?
                .checked_add(*new_length)?
        } else {
            field.end - field.start
        };
        sum.checked_add(size)
    });
    let graph_length = graph_length.context("图长度溢出")?;
    {
        let mut dst = guarded_new_file(&output, &output_root, false)?;
        for field in &original.top {
            if field.number != 7 {
                copy_span(&mut src, field.start, field.end, &mut dst)?;
                continue;
            }
            dst.write_all(&varint(7 * 8 + 2))?;
            dst.write_all(&varint(graph_length))?;
            for item in &original.graph_fields {
                let Some((_, _, remaining, extra, new_length, _)) =
                    planned.iter().find(|(start, ..)| *start == item.start)
                else {
                    copy_span(&mut src, item.start, item.end, &mut dst)?;
                    continue;
                };
                dst.write_all(&varint(5 * 8 + 2))?;
                dst.write_all(&varint(*new_length))?;
                for inner in remaining {
                    copy_span(&mut src, inner.start, inner.end, &mut dst)?;
                }
                dst.write_all(extra)?;
            }
        }
    }
    let mut derived_file = File::open(&output)?;
    let derived_len = fs::symlink_metadata(&output)?.len();
    let derived = inspect_file(&mut derived_file, derived_len)?;
    for (left, right, label) in [
        (
            &original.initializer_metadata_sha256,
            &derived.initializer_metadata_sha256,
            "initializer_metadata_sha256",
        ),
        (
            &original.other_graph_fields_sha256,
            &derived.other_graph_fields_sha256,
            "other_graph_fields_sha256",
        ),
        (
            &original.other_model_fields_sha256,
            &derived.other_model_fields_sha256,
            "other_model_fields_sha256",
        ),
    ] {
        ensure!(left == right, "语义元数据发生变化：{label}");
    }
    let derived_names = derived
        .tensors
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect::<BTreeSet<_>>();
    let source_names = original
        .tensors
        .iter()
        .map(|tensor| tensor.name.as_str())
        .collect::<BTreeSet<_>>();
    ensure!(derived_names == source_names, "initializer 名称发生变化");
    let copied = identities
        .iter()
        .filter_map(|item| Some((item.get("name")?.as_str()?.to_owned(), item.clone())))
        .collect::<BTreeMap<_, _>>();
    let mut verified = Vec::new();
    let mut weights = File::open(&weights_path)?;
    for tensor in &original.tensors {
        let candidate = derived
            .tensors
            .iter()
            .find(|item| item.name == tensor.name)
            .context("缺少导出的 initializer")?;
        ensure!(
            tensor.dtype == candidate.dtype && tensor.dimensions == candidate.dimensions,
            "initializer 的 dtype 或维度发生变化"
        );
        let mut source_payload = None;
        let mut candidate_payload = None;
        if let Some(raw) = tensor.raw_field {
            if let Some(block) = copied.get(&tensor.name) {
                source_payload = block
                    .get("payload_sha256")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                let offset = block
                    .get("offset")
                    .and_then(Value::as_u64)
                    .context("偏移")?;
                let bytes = block.get("bytes").and_then(Value::as_u64).context("长度")?;
                let mut expected = BTreeMap::new();
                expected.insert("location".to_owned(), weights_name.clone());
                expected.insert("offset".to_owned(), offset.to_string());
                expected.insert("length".to_owned(), bytes.to_string());
                ensure!(
                    candidate.external == expected && candidate.data_location == 1,
                    "外置权重位置与计划不一致"
                );
                candidate_payload = Some(hash_span(&mut weights, offset, offset + bytes)?);
            } else {
                source_payload = Some(hash_span(&mut src, raw.payload, raw.end)?);
                let raw = candidate.raw_field.context("内嵌权重丢失了 raw_data")?;
                candidate_payload = Some(hash_span(&mut derived_file, raw.payload, raw.end)?);
            }
            ensure!(
                source_payload == candidate_payload,
                "写出的权重与源字节不一致"
            );
        }
        verified.push(json!({
            "name": tensor.name,
            "dtype": tensor.dtype,
            "dimensions": tensor.dimensions,
            "payload_bytes": tensor.raw_bytes,
            "source_payload_sha256": source_payload,
            "candidate_payload_sha256": candidate_payload,
            "dtype_dimensions_payload_equal": true,
            "representation": if copied.contains_key(&tensor.name) { "external" } else { "unchanged_inline" },
        }));
    }
    drop(src);
    drop(weights);
    drop(derived_file);
    let after = file_stamp(&fs::symlink_metadata(&source)?)?;
    ensure!(
        before == after && sha256_file(&source)? == source_sha,
        "准备过程中源文件发生变化"
    );
    let receipt = json!({
        "schema": 1,
        "prepared": true,
        "sessions_created": 0,
        "inference_calls": 0,
        "source": source,
        "candidate": output,
        "source_sha256": source_sha,
        "output_root": output_root,
        "candidate_sha256": sha256_file(&output)?,
        "external_weights_sha256": sha256_file(&weights_path)?,
        "source_summary": summary(&original),
        "candidate_summary": summary(&derived),
        "payloads_byte_identical": true,
        "metadata_byte_identical": true,
        "externalized": identities,
        "tensor_verification": verified,
        "verified_tensor_count": original.tensors.len(),
        "external_bytes": total,
        "alignment": options.alignment,
        "execution_state": "prepared_only_no_runtime_session",
    });
    let mut text = serde_json::to_string_pretty(&receipt)?;
    text.push('\n');
    guarded_new_file(&receipt_path, &output_root, true)?.write_all(text.as_bytes())?;
    Ok(receipt)
}

pub(crate) fn onnx_external_sidecar_ready(onnx: &Path, weights_name: &str) -> Result<bool> {
    let onnx = match checked_source(onnx) {
        Ok(path) => path,
        Err(_) => return Ok(false),
    };
    let weights = onnx.with_file_name(weights_name);
    let weights_meta = match fs::symlink_metadata(&weights) {
        Ok(meta) if meta.is_file() && !is_reparse(&meta) && meta.len() > 0 => meta,
        _ => return Ok(false),
    };
    let mut file = File::open(&onnx)?;
    let view = match inspect_file(&mut file, fs::symlink_metadata(&onnx)?.len()) {
        Ok(view) => view,
        Err(_) => return Ok(false),
    };
    let mut externalized = 0usize;
    for tensor in &view.tensors {
        if tensor.raw_bytes >= 4096 && tensor.external.is_empty() {
            return Ok(false);
        }
        if tensor.external.is_empty() {
            continue;
        }
        if tensor.data_location != 1
            || tensor.external.get("location").map(String::as_str) != Some(weights_name)
        {
            return Ok(false);
        }
        let offset: u64 = tensor
            .external
            .get("offset")
            .and_then(|value| value.parse().ok())
            .unwrap_or(u64::MAX);
        let length: u64 = tensor
            .external
            .get("length")
            .and_then(|value| value.parse().ok())
            .unwrap_or(u64::MAX);
        if offset
            .checked_add(length)
            .is_none_or(|end| end > weights_meta.len())
        {
            return Ok(false);
        }
        externalized += 1;
    }
    Ok(externalized > 0)
}

fn summary(view: &Inspection) -> Value {
    let mut raw_bytes = Map::new();
    for tensor in &view.tensors {
        let key = tensor.dtype.to_string();
        let current = raw_bytes.get(&key).and_then(Value::as_u64).unwrap_or(0);
        raw_bytes.insert(key, json!(current + tensor.raw_bytes));
    }
    let mut external_files = view
        .tensors
        .iter()
        .filter_map(|tensor| tensor.external.get("location").cloned())
        .collect::<Vec<_>>();
    external_files.sort();
    external_files.dedup();
    json!({
        "bytes": view.bytes,
        "initializer_metadata_sha256": view.initializer_metadata_sha256,
        "other_graph_fields_sha256": view.other_graph_fields_sha256,
        "other_model_fields_sha256": view.other_model_fields_sha256,
        "initializer_count": view.tensors.len(),
        "raw_bytes_by_dtype": raw_bytes,
        "external_initializer_count": view.tensors.iter().filter(|tensor| !tensor.external.is_empty()).count(),
        "external_files": external_files,
    })
}

fn inspect_file(stream: &mut File, len: u64) -> Result<Inspection> {
    let top = read_fields(stream, 0, len)?;
    let graphs = top
        .iter()
        .copied()
        .filter(|field| field.number == 7)
        .collect::<Vec<_>>();
    ensure!(
        graphs.len() == 1 && graphs[0].wire == 2,
        "模型应只有一个 graph"
    );
    let graph = graphs[0];
    let graph_fields = read_fields(stream, graph.payload, graph.end)?;
    let mut initializer_meta = Sha256::new();
    let mut other_graph = Sha256::new();
    let mut other_model = Sha256::new();
    for field in &top {
        if field.number != 7 {
            update_span(stream, field.start, field.end, &mut other_model)?;
        }
    }
    let mut tensors = Vec::new();
    let mut seen = BTreeSet::new();
    for field in &graph_fields {
        if field.number != 5 {
            update_span(stream, field.start, field.end, &mut other_graph)?;
            continue;
        }
        let inner = read_fields(stream, field.payload, field.end)?;
        let names = inner
            .iter()
            .copied()
            .filter(|item| item.number == 8 && item.wire == 2)
            .collect::<Vec<_>>();
        ensure!(names.len() == 1, "initializer 名称缺失或重复");
        let name = String::from_utf8(small_bytes(stream, &names[0])?)
            .context("initializer 名称不是 UTF-8")?;
        ensure!(seen.insert(name.clone()), "initializer 名称重复");
        let dtypes = inner
            .iter()
            .filter_map(|item| (item.number == 2 && item.wire == 0).then_some(item.scalar))
            .collect::<Vec<_>>();
        ensure!(dtypes.len() == 1, "initializer 缺少 dtype");
        let dtype = dtypes[0].context("dtype")?;
        let raw = inner
            .iter()
            .copied()
            .filter(|item| item.number == 9)
            .collect::<Vec<_>>();
        ensure!(
            raw.len() <= 1 && raw.iter().all(|item| item.wire == 2),
            "raw_data 无效"
        );
        let mut tensor_meta = Sha256::new();
        let mut dimensions = Vec::new();
        let mut external = BTreeMap::new();
        let mut location = 0u64;
        for item in &inner {
            if item.number == 1 {
                if item.wire == 0 {
                    dimensions.push(item.scalar.context("维度")?);
                } else if item.wire == 2 {
                    let packed = small_bytes(stream, item)?;
                    let mut position = 0;
                    while position < packed.len() {
                        let (dimension, next) = read_bytes_varint(&packed, position)?;
                        dimensions.push(dimension);
                        position = next;
                    }
                } else {
                    bail!("维度字段无效");
                }
            }
            if item.number == 14 {
                ensure!(item.wire == 0, "data_location 无效");
                location = item.scalar.context("data_location")?;
            } else if item.number == 13 {
                let entries = read_fields(stream, item.payload, item.end)?;
                let mut entry = BTreeMap::new();
                for part in &entries {
                    let text = String::from_utf8(small_bytes(stream, part)?)
                        .context("external_data 不是 UTF-8")?;
                    ensure!(
                        entry.insert(part.number, text).is_none(),
                        "external_data 项重复"
                    );
                }
                ensure!(
                    entry.len() == 2 && entry.contains_key(&1) && entry.contains_key(&2),
                    "external_data 项无效"
                );
                let key = entry.remove(&1).unwrap();
                let value = entry.remove(&2).unwrap();
                ensure!(
                    external.insert(key, value).is_none(),
                    "external_data 键重复"
                );
            } else if item.number != 9 {
                update_span(stream, item.start, item.end, &mut tensor_meta)?;
            }
        }
        let digest = tensor_meta.finalize_reset();
        let encoded = name.as_bytes();
        initializer_meta.update(varint(encoded.len() as u64));
        initializer_meta.update(encoded);
        initializer_meta.update(&digest);
        tensors.push(TensorView {
            name,
            dtype,
            dimensions,
            raw_bytes: raw
                .first()
                .map(|field| field.end - field.payload)
                .unwrap_or(0),
            data_location: location,
            external,
            field: *field,
            inner,
            raw_field: raw.first().copied(),
            metadata_sha256: hex(&digest),
        });
    }
    Ok(Inspection {
        bytes: len,
        graph_fields,
        top,
        tensors,
        initializer_metadata_sha256: hex(&initializer_meta.finalize()),
        other_graph_fields_sha256: hex(&other_graph.finalize()),
        other_model_fields_sha256: hex(&other_model.finalize()),
    })
}

fn external_fields(filename: &str, offset: u64, length: u64) -> Vec<u8> {
    let entries = [
        ("location", filename.to_owned()),
        ("offset", offset.to_string()),
        ("length", length.to_string()),
    ];
    let mut out = Vec::new();
    for (key, value) in entries {
        let body = {
            let mut body = length_field(1, key.as_bytes());
            body.extend(length_field(2, value.as_bytes()));
            body
        };
        out.extend(length_field(13, &body));
    }
    out.extend(varint(14 * 8));
    out.extend(varint(1));
    out
}

fn varint(mut value: u64) -> Vec<u8> {
    let mut output = Vec::new();
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value == 0 {
            output.push(byte);
            break;
        }
        byte |= 0x80;
        output.push(byte);
    }
    output
}

fn length_field(number: u64, value: &[u8]) -> Vec<u8> {
    let mut out = varint(number * 8 + 2);
    out.extend(varint(value.len() as u64));
    out.extend_from_slice(value);
    out
}

fn read_varint(stream: &mut File) -> Result<u64> {
    let mut result = 0u64;
    let mut shift = 0;
    while shift < 64 {
        let mut byte = [0u8; 1];
        stream
            .read_exact(&mut byte)
            .context("截断的 protobuf varint")?;
        result |= u64::from(byte[0] & 0x7f) << shift;
        if byte[0] < 128 {
            return Ok(result);
        }
        shift += 7;
    }
    bail!("protobuf varint 超出范围")
}

fn read_bytes_varint(data: &[u8], mut position: usize) -> Result<(u64, usize)> {
    let mut value = 0u64;
    let mut shift = 0;
    while position < data.len() && shift < 64 {
        let byte = data[position];
        position += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte < 128 {
            return Ok((value, position));
        }
        shift += 7;
    }
    bail!("截断或无效的 protobuf varint")
}

fn read_fields(stream: &mut File, start: u64, end: u64) -> Result<Vec<Field>> {
    let mut position = start;
    let mut fields = Vec::new();
    while position < end {
        stream.seek(SeekFrom::Start(position))?;
        let tag = read_varint(stream)?;
        let number = tag >> 3;
        let wire = (tag & 7) as u8;
        ensure!(number != 0, "protobuf 字段号为零");
        let (scalar, payload, finish) = match wire {
            0 => {
                let scalar = read_varint(stream)?;
                let at = stream.stream_position()?;
                (Some(scalar), at, at)
            }
            2 => {
                let size = read_varint(stream)?;
                let payload = stream.stream_position()?;
                (
                    None,
                    payload,
                    payload.checked_add(size).context("字段长度溢出")?,
                )
            }
            1 => {
                let payload = stream.stream_position()?;
                (None, payload, payload + 8)
            }
            5 => {
                let payload = stream.stream_position()?;
                (None, payload, payload + 4)
            }
            _ => bail!("不支持的 protobuf wire 类型"),
        };
        ensure!(finish <= end, "protobuf 字段超出消息范围");
        fields.push(Field {
            number,
            wire,
            start: position,
            payload,
            end: finish,
            scalar,
        });
        position = finish;
    }
    Ok(fields)
}

fn small_bytes(stream: &mut File, field: &Field) -> Result<Vec<u8>> {
    let len = field.end - field.payload;
    ensure!(len <= 65536, "元数据字段过大");
    let mut buffer = vec![0u8; len as usize];
    stream.seek(SeekFrom::Start(field.payload))?;
    stream.read_exact(&mut buffer)?;
    Ok(buffer)
}

fn update_span(stream: &mut File, start: u64, end: u64, hasher: &mut Sha256) -> Result<()> {
    copy_span(stream, start, end, hasher)
}

fn hash_span(stream: &mut File, start: u64, end: u64) -> Result<String> {
    let mut hasher = Sha256::new();
    update_span(stream, start, end, &mut hasher)?;
    Ok(hex(&hasher.finalize()))
}

fn copy_span(stream: &mut File, mut start: u64, end: u64, output: &mut dyn Write) -> Result<()> {
    stream.seek(SeekFrom::Start(start))?;
    let mut buffer = vec![0u8; CHUNK as usize];
    while start < end {
        let want = ((end - start).min(CHUNK)) as usize;
        let n = stream.read(&mut buffer[..want])?;
        ensure!(n > 0, "文件被截断");
        output.write_all(&buffer[..n])?;
        start += n as u64;
    }
    Ok(())
}

struct HashWriter<'a, W> {
    inner: &'a mut W,
    hasher: &'a mut Sha256,
}
impl<W: Write> Write for HashWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

fn write_zeros(output: &mut impl Write, mut count: u64) -> Result<()> {
    let zeros = [0u8; 4096];
    while count > 0 {
        let n = (count as usize).min(zeros.len());
        output.write_all(&zeros[..n])?;
        count -= n as u64;
    }
    Ok(())
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    let mask = alignment - 1;
    value
        .checked_add(mask)
        .map(|sum| sum & !mask)
        .context("对齐溢出")
}

fn output_paths(
    source: &Path,
    output: &Path,
    output_root: &Path,
) -> Result<(PathBuf, PathBuf, PathBuf, PathBuf, PathBuf)> {
    let source = checked_source(source)?;
    let output_root = assert_no_reparse(output_root)?;
    let output = assert_no_reparse(output)?;
    match fs::symlink_metadata(&output_root) {
        Ok(meta) if meta.is_dir() => {}
        Ok(_) => bail!("输出根目录不是文件夹"),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let extension_ok = output
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("onnx"));
    ensure!(
        output != output_root && is_within(&output, &output_root) && extension_ok,
        "导出的 ONNX 必须位于指定输出目录内"
    );
    let source_dir = source.parent().context("源目录")?;
    ensure!(
        !is_within(source_dir, &output_root) && !is_within(&output_root, source_dir),
        "源目录与输出目录不能重叠"
    );
    let weights = output.with_extension("weights.bin");
    let receipt = output.with_extension("external.json");
    for path in [&output, &weights, &receipt] {
        assert_no_reparse(path)?;
        match fs::symlink_metadata(path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Ok(_) => bail!("拒绝覆盖已有模型、权重或收据"),
            Err(error) => return Err(error.into()),
        }
    }
    Ok((source, output, output_root, weights, receipt))
}

fn guarded_new_file(path: &Path, output_root: &Path, _text: bool) -> Result<File> {
    let output_root = assert_no_reparse(output_root)?;
    let path = assert_no_reparse(path)?;
    ensure!(is_within(&path, &output_root), "输出超出了声明的根目录");
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .with_context(|| format!("创建 {}", path.display()))
}

fn checked_source(path: &Path) -> Result<PathBuf> {
    let path = assert_no_reparse(path)?;
    let meta = fs::symlink_metadata(&path)?;
    ensure!(meta.is_file(), "源模型必须是已存在的普通文件");
    Ok(path)
}

pub(crate) fn assert_no_reparse(path: &Path) -> Result<PathBuf> {
    let path = absolute_lexical(path)?;
    let mut cursor = path.clone();
    loop {
        match fs::symlink_metadata(&cursor) {
            Ok(meta) if is_reparse(&meta) => bail!("不允许重解析或符号链接：{}", cursor.display()),
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        if !cursor.pop() {
            break;
        }
    }
    Ok(path)
}

fn absolute_lexical(path: &Path) -> Result<PathBuf> {
    for component in path.components() {
        ensure!(
            !matches!(component, Component::ParentDir),
            "不允许父目录穿越"
        );
    }
    reject_windows_shape(path)?;
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn reject_windows_shape(path: &Path) -> Result<()> {
    let text = path.to_string_lossy();
    let bytes = text.as_bytes();
    if bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':' {
        let rest = &bytes[2..];
        ensure!(
            rest.first()
                .is_some_and(|byte| *byte == b'/' || *byte == b'\\'),
            "不允许盘符相对路径"
        );
    }
    for component in path.components() {
        if matches!(component, Component::Prefix(_) | Component::RootDir) {
            continue;
        }
        let part = component.as_os_str().to_string_lossy();
        ensure!(!part.contains(':'), "不允许备用数据流或盘符相对路径");
        ensure!(
            !part.ends_with(' ') && !part.ends_with('.'),
            "不允许含混的 Windows 路径分量"
        );
        ensure!(
            !part
                .chars()
                .any(|ch| matches!(ch, '<' | '>' | '"' | '|' | '?' | '*')),
            "路径包含 Windows 非法字符"
        );
        let stem = part.split('.').next().unwrap_or(&part).to_ascii_uppercase();
        let reserved = matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
            || ((stem.starts_with("COM") || stem.starts_with("LPT"))
                && stem.len() == 4
                && stem.as_bytes()[3].is_ascii_digit()
                && stem.as_bytes()[3] != b'0');
        ensure!(!reserved, "路径包含 Windows 保留名");
    }
    Ok(())
}

fn is_within(path: &Path, root: &Path) -> bool {
    let path = normalized_components(path);
    let root = normalized_components(root);
    path.len() >= root.len() && path[..root.len()] == root[..]
}

fn normalized_components(path: &Path) -> Vec<String> {
    path.components()
        .map(|component| {
            let text = component.as_os_str().to_string_lossy().into_owned();
            #[cfg(windows)]
            {
                text.to_ascii_lowercase()
            }
            #[cfg(not(windows))]
            text
        })
        .collect()
}

fn is_reparse(meta: &fs::Metadata) -> bool {
    if meta.file_type().is_symlink() {
        return true;
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::MetadataExt;
        return meta.file_attributes() & 0x400 != 0;
    }
    #[cfg(not(windows))]
    false
}

fn modified_nanos(meta: &fs::Metadata) -> u128 {
    meta.modified()
        .ok()
        .and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
        .unwrap_or(0)
}

fn file_stamp(meta: &fs::Metadata) -> Result<(u64, u64, u64, u128)> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        return Ok((meta.dev(), meta.ino(), meta.len(), modified_nanos(meta)));
    }
    #[cfg(windows)]
    {
        // file_index is still unstable. Size plus mtime catches replacement;
        // the following SHA256 comparison checks the bytes.
        return Ok((0, 0, meta.len(), modified_nanos(meta)));
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = meta;
        bail!("不支持的系统")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(
        name: &str,
        dtype: u64,
        payload: Option<&[u8]>,
        dimensions: &[u64],
        typed: Option<&[u64]>,
    ) -> Vec<u8> {
        let mut value = length_field(8, name.as_bytes());
        value.extend(varint(16));
        value.extend(varint(dtype));
        let mut packed = Vec::new();
        for dimension in dimensions {
            packed.extend(varint(*dimension));
        }
        value.extend(length_field(1, &packed));
        if let Some(payload) = payload {
            value.extend(length_field(9, payload));
        }
        if let Some(typed) = typed {
            let mut body = Vec::new();
            for item in typed {
                body.extend(varint(*item));
            }
            value.extend(length_field(7, &body));
        }
        value.extend(length_field(12, b"unchanged documentation"));
        value
    }

    fn synthetic_model() -> Vec<u8> {
        let fp16: Vec<u8> = (0..128).collect();
        let quantized: Vec<u8> = (0..65).rev().collect();
        let tensors = [
            tensor("fp16", 10, Some(&fp16), &[64], None),
            tensor("uint8-packed-weight", 2, Some(&quantized), &[65], None),
            tensor("control", 7, Some(b"12345678"), &[], None),
            tensor("typed-constant", 7, None, &[2], Some(&[3, 5])),
        ];
        let mut graph = length_field(1, &length_field(4, b"Identity"));
        for item in tensors {
            graph.extend(length_field(5, &item));
        }
        graph.extend(length_field(2, b"unchanged graph name"));
        let mut model = varint(8);
        model.extend(varint(10));
        model.extend(length_field(7, &graph));
        model.extend(length_field(2, b"producer"));
        model
    }

    fn temp(label: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "yyur-ext-{label}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn rust_sidecar_matches_python_tool_on_synthetic_onnx() {
        let root = temp("compare");
        let source_dir = root.join("inputs");
        fs::create_dir_all(&source_dir).unwrap();
        let source = source_dir.join("source.onnx");
        let original = synthetic_model();
        fs::write(&source, &original).unwrap();
        let rust_root = root.join("rust-out");
        let python_root = root.join("python-out");
        let options = PrepareOptions {
            minimum: 32,
            alignment: 256,
            expected_source_sha256: Some(sha256_file(&source).unwrap()),
        };
        let receipt =
            prepare(&source, &rust_root.join("model.onnx"), &rust_root, &options).unwrap();
        assert_eq!(receipt["sessions_created"], 0);
        assert_eq!(receipt["inference_calls"], 0);
        assert_eq!(receipt["external_bytes"], 321);
        assert_eq!(receipt["verified_tensor_count"], 4);
        let fp16: Vec<u8> = (0..128).collect();
        let quantized: Vec<u8> = (0..65).rev().collect();
        let mut expected_weights = fp16;
        expected_weights.extend(std::iter::repeat_n(0u8, 128));
        expected_weights.extend(quantized);
        assert_eq!(
            fs::read(rust_root.join("model.weights.bin")).unwrap(),
            expected_weights
        );
        let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tools/models/prepare_external_weights.py");
        let mut cmd = find_python_command();
        let status = cmd
            .arg(&script)
            .arg("--source")
            .arg(&source)
            .arg("--output")
            .arg(python_root.join("model.onnx"))
            .arg("--output-root")
            .arg(&python_root)
            .arg("--minimum")
            .arg("32")
            .arg("--alignment")
            .arg("256")
            .arg("--expected-source-sha256")
            .arg(sha256_file(&source).unwrap())
            .status()
            .expect("python");
        assert!(status.success(), "python prepare failed");
        assert_eq!(
            fs::read(rust_root.join("model.onnx")).unwrap(),
            fs::read(python_root.join("model.onnx")).unwrap()
        );
        assert_eq!(
            fs::read(rust_root.join("model.weights.bin")).unwrap(),
            fs::read(python_root.join("model.weights.bin")).unwrap()
        );
        assert_eq!(fs::read(&source).unwrap(), original);
        assert!(
            onnx_external_sidecar_ready(&rust_root.join("model.onnx"), "model.weights.bin")
                .unwrap()
        );
        assert!(!onnx_external_sidecar_ready(&source, "model.weights.bin").unwrap());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn wrong_hash_overlap_and_traversal_create_nothing() {
        let root = temp("reject");
        let source = root.join("inputs/source.onnx");
        fs::create_dir_all(source.parent().unwrap()).unwrap();
        fs::write(&source, synthetic_model()).unwrap();
        let output_root = root.join("outputs");
        let output = output_root.join("model.onnx");
        let options = PrepareOptions {
            minimum: 32,
            alignment: 256,
            expected_source_sha256: Some("0".repeat(64)),
        };
        assert!(prepare(&source, &output, &output_root, &options).is_err());
        assert!(!output_root.exists());
        assert!(
            prepare(
                &source,
                &root.join("elsewhere/model.onnx"),
                &output_root,
                &PrepareOptions {
                    minimum: 32,
                    alignment: 256,
                    expected_source_sha256: None,
                }
            )
            .is_err()
        );
        assert!(
            prepare(
                &source.parent().unwrap().join("../inputs/source.onnx"),
                &output,
                &output_root,
                &PrepareOptions {
                    minimum: 32,
                    alignment: 256,
                    expected_source_sha256: None,
                }
            )
            .is_err()
        );
        assert!(
            prepare(
                &source,
                &source,
                &source.parent().unwrap(),
                &PrepareOptions {
                    minimum: 32,
                    alignment: 256,
                    expected_source_sha256: None,
                }
            )
            .is_err()
        );
        assert!(!output_root.exists());
        assert!(reject_windows_shape(Path::new("C:relative.onnx")).is_err());
        assert!(reject_windows_shape(Path::new("folder/CON.onnx")).is_err());
        assert!(reject_windows_shape(Path::new("folder/file:stream.onnx")).is_err());
        #[cfg(unix)]
        {
            let link = root.join("link.onnx");
            std::os::unix::fs::symlink(&source, &link).unwrap();
            assert!(
                prepare(
                    &link,
                    &output,
                    &output_root,
                    &PrepareOptions {
                        minimum: 32,
                        alignment: 256,
                        expected_source_sha256: None,
                    }
                )
                .is_err()
            );
        }
        let _ = fs::remove_dir_all(root);
    }

    fn find_python_command() -> std::process::Command {
        let candidates: &[(&str, &[&str])] = if cfg!(windows) {
            &[("python", &[]), ("py", &["-3"]), ("python3", &[])]
        } else {
            &[("python3", &[]), ("python", &[])]
        };
        for (exe, prefix) in candidates {
            let ok = std::process::Command::new(exe)
                .args(*prefix)
                .arg("--version")
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .is_ok_and(|s| s.success());
            if ok {
                let mut cmd = std::process::Command::new(exe);
                cmd.args(*prefix);
                return cmd;
            }
        }
        panic!("no working Python 3 executable found in PATH");
    }
}
