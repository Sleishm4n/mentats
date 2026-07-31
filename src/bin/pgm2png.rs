use std::env;
use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: cargo run --bin pgm2png <input.pgm> [output.png]");
        std::process::exit(1);
    }

    let input_path = Path::new(&args[1]);
    let output_path = if args.len() >= 3 {
        args[2].clone()
    } else {
        input_path
            .with_extension("png")
            .to_string_lossy()
            .to_string()
    };

    let (width, height, max_val, pixels) = parse_pgm(input_path)?;
    save_png(&output_path, width, height, max_val, &pixels)?;

    println!("Converted: {} -> {}", input_path.display(), output_path);
    Ok(())
}

fn parse_pgm(path: &Path) -> Result<(u32, u32, u32, Vec<u8>), Box<dyn std::error::Error>> {
    let mut file = File::open(path)?;
    let mut content = String::new();
    file.read_to_string(&mut content)?;

    // Clean comments
    let clean_lines: Vec<String> = content
        .lines()
        .map(|line| line.split('#').next().unwrap_or("").trim().to_string())
        .filter(|line| !line.is_empty())
        .collect();

    if clean_lines.is_empty() {
        return Err("Empty PGM file".into());
    }

    let mut line_iter = clean_lines.into_iter();

    // Parse header
    let format = line_iter.next().ok_or("Missing PGM format header")?;
    if format != "P2" {
        return Err(format!("Unsupported format '{}' (expected P2 ASCII)", format).into());
    }

    let header_dims = line_iter.next().ok_or("Missing dimensions line")?;
    let dims: Vec<&str> = header_dims.split_whitespace().collect();
    let width: u32 = dims[0].parse()?;
    let height: u32 = dims[1].parse()?;

    let max_val: u32 = line_iter.next().ok_or("Missing max value")?.parse()?;

    let total_pixels = (width * height) as usize;
    let mut pixels = Vec::with_capacity(total_pixels);

    // Read pixel rows
    for line in line_iter {
        // Try space-separated first
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() > 1 {
            for tok in tokens {
                if let Ok(v) = tok.parse::<u32>() {
                    let norm = ((v as f32 / max_val as f32) * 255.0).round() as u8;
                    pixels.push(norm);
                }
            }
        } else if !line.is_empty() {
            for token in line.split_whitespace() {
                if let Ok(v) = token.parse::<u32>() {
                    let norm = ((v as f32 / max_val as f32) * 255.0).round() as u8;
                    pixels.push(norm);
                }
            }
        }
    }

    if pixels.len() < total_pixels {
        return Err(format!(
            "Unexpected EOF: Expected {} pixels, found {}",
            total_pixels,
            pixels.len()
        )
        .into());
    }

    Ok((width, height, max_val, pixels))
}

fn save_png(
    path: &str,
    width: u32,
    height: u32,
    _max_val: u32,
    pixels: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // PNG Header
    writer.write_all(&[137, 80, 78, 71, 13, 10, 26, 10])?;

    // IHDR Chunk
    let mut ihdr_data = Vec::new();
    ihdr_data.extend_from_slice(&width.to_be_bytes());
    ihdr_data.extend_from_slice(&height.to_be_bytes());
    ihdr_data.push(8); // 8-bit depth
    ihdr_data.push(0); // Grayscale
    ihdr_data.push(0);
    ihdr_data.push(0);
    ihdr_data.push(0);
    write_png_chunk(&mut writer, b"IHDR", &ihdr_data)?;

    // IDAT Chunk
    let mut uncompressed = Vec::with_capacity(((width + 1) * height) as usize);
    for row in 0..height {
        uncompressed.push(0); // Filter type 0
        let start = (row * width) as usize;
        let end = start + width as usize;
        uncompressed.extend_from_slice(&pixels[start..end]);
    }

    let compressed = miniz_deflate(&uncompressed);
    write_png_chunk(&mut writer, b"IDAT", &compressed)?;

    // IEND Chunk
    write_png_chunk(&mut writer, b"IEND", &[])?;

    Ok(())
}

fn write_png_chunk(writer: &mut impl Write, chunk_type: &[u8; 4], data: &[u8]) -> io::Result<()> {
    writer.write_all(&(data.len() as u32).to_be_bytes())?;
    writer.write_all(chunk_type)?;
    writer.write_all(data)?;

    let mut crc = 0xffffffffu32;
    crc = update_crc(crc, chunk_type);
    crc = update_crc(crc, data);
    writer.write_all(&(crc ^ 0xffffffffu32).to_be_bytes())?;

    Ok(())
}

fn update_crc(mut crc: u32, data: &[u8]) -> u32 {
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if (crc & 1) != 0 {
                crc = (crc >> 1) ^ 0xedb88320;
            } else {
                crc >>= 1;
            }
        }
    }
    crc
}

fn miniz_deflate(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(0x78);
    out.push(0x01);

    let max_block = 65535;
    let chunks: Vec<&[u8]> = data.chunks(max_block).collect();

    for (i, chunk) in chunks.iter().enumerate() {
        let is_last = i == chunks.len() - 1;
        out.push(if is_last { 0x01 } else { 0x00 });

        let len = chunk.len() as u16;
        let nlen = !len;

        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&nlen.to_le_bytes());
        out.extend_from_slice(chunk);
    }

    let mut s1 = 1u32;
    let mut s2 = 0u32;
    for &b in data {
        s1 = (s1 + b as u32) % 65521;
        s2 = (s2 + s1) % 65521;
    }
    let adler32 = (s2 << 16) | s1;
    out.extend_from_slice(&adler32.to_be_bytes());

    out
}
