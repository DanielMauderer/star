#![feature(portable_simd)]

use futures::executor::block_on;
use generator::{gpu, sequ, simd};

const GALAXY_RADIUS: f32 = 1. / 30003.2615637769;
const GALAXY_HEIGHT: f32 = 1. / 2503.2615637769;
const NUM_POINTS: usize = 100_000_000;
mod generator;

fn main() {
    println!("go");
    let done = match block_on(gpu::run()) {
        Ok(d) => d,
        Err(e) => {
            println!("error: {}", e);
            return;
        }
    };
    println!("gpu processing time: {:?}", done);

    let done = simd::run();
    println!("simd processing time: {:?}", done);

    let done = sequ::run();
    println!("unoptimized processing time: {:?}", done);
}
