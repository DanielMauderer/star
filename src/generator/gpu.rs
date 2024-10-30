use crate::{GALAXY_HEIGHT, GALAXY_RADIUS, NUM_POINTS};
use ocl::{Buffer, ProQue};
use std::{io::Write, time::Instant};
use turborand::prelude::*;

extern crate ocl;
extern crate ocl_extras;

pub async fn run() -> ocl::Result<std::time::Duration> {
    let duration = std::time::Instant::now();
    let src = include_str!("generate_x.cl");

    let pro_que = ProQue::builder().src(src).dims(NUM_POINTS).build()?;

    println!("{:?} - Creating a source buffer...", duration.elapsed());

    let mut points: Vec<f32> = vec![0.0; NUM_POINTS * 3];
    fill_points(&pro_que, &mut points, &duration)?;

    Ok(duration.elapsed())
}

fn fill_points(
    pro_que: &ProQue,
    result_source: &mut Vec<f32>,
    duration: &Instant,
) -> ocl::Result<()> {
    println!("(x): {:?} - Creating GPU buffers...", duration.elapsed());

    let rand = Rng::new();
    let seed_count = NUM_POINTS / 100;
    let mut seed_source = vec![0i64; seed_count];

    for i in 0..seed_count {
        seed_source[i] = rand.gen_i64();
    }

    let seed_buffer: Buffer<i64> = unsafe {
        Buffer::<i64>::builder()
            .queue(pro_que.queue().clone())
            .len(seed_count)
            .use_host_slice(&seed_source)
            .build()
    }?;

    let result_buffer: Buffer<f32> = Buffer::<f32>::builder()
        .queue(pro_que.queue().clone())
        .len(NUM_POINTS * 3)
        .fill_val(Default::default())
        .build()?;

    println!("(x): {:?} - Creating kernel...", duration.elapsed());

    let kernel = pro_que
        .kernel_builder("generate_points")
        .global_work_size(NUM_POINTS)
        .arg(&seed_buffer)
        .arg(&result_buffer)
        .arg(GALAXY_HEIGHT)
        .arg(GALAXY_RADIUS)
        .build()?;
    println!("(x): {:?} - Enqueueing kernel...", duration.elapsed());

    unsafe {
        kernel.enq()?;
    }
    println!("(x): {:?} - Reading results...", duration.elapsed());
    result_buffer.read(result_source).enq()?;
    Ok(())
}

async fn write_batch_to_file(
    writer: &mut std::io::BufWriter<std::fs::File>,
    points: Vec<f32>,
) -> std::io::Result<()> {
    for j in 0..NUM_POINTS {
        writer.write_all(&points[j].to_ne_bytes())?;
    }
    writer.flush()?;
    Ok(())
}
