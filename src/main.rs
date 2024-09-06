#![feature(portable_simd)]

use generator::{gpu, simd, unoptimized};

const GALAXY_RADIUS: f32 = 1. / 30003.2615637769;
const GALAXY_HEIGHT: f32 = 1. / 2503.2615637769;
const NUM_POINTS: usize = 100_000_000;
mod generator;

fn main() {
    println!("go");
    let done = gpu::run();
    println!("gpu processing time: {:?}", done);

    let done = simd::run();
    println!("simd processing time: {:?}", done);
    let done = unoptimized::run();
    println!("unoptimized processing time: {:?}", done);
}
