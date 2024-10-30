// x sample took 1.624668826s
// y sample took 1.905179329s
// z sample took 195.117154ms
// cartesian conversion took 714.092538ms

use std::{
    simd::{f32x4, f32x64, Mask, StdFloat},
    time::{Duration, Instant},
};

use rand::{thread_rng, Fill};
use rand_distr::num_traits::ToPrimitive;

use crate::{GALAXY_HEIGHT, GALAXY_RADIUS, NUM_POINTS};

#[allow(dead_code)]
pub fn run() -> Duration {
    let mut polar_points_x = vec![f32x64::splat(0.); NUM_POINTS / 64];
    let mut polar_points_y = vec![f32x64::splat(0.); NUM_POINTS / 64];
    let mut polar_points_z = vec![f32x64::splat(0.); NUM_POINTS / 64];

    let simd_radius_lambda = f32x64::splat(GALAXY_RADIUS);
    let simd_height_lambda = f32x64::splat(GALAXY_HEIGHT);

    let mut rng = thread_rng();

    let now = Instant::now();

    let mut start_offset = now.elapsed();

    let mut buffer = [0.; 64];
    for n in 0..NUM_POINTS / 64 {
        buffer.try_fill(&mut rng).unwrap();
        polar_points_x[n] = (-f32x64::from_array(buffer).ln() / simd_radius_lambda).cos();
    }

    println!("x sample took {:?}", now.elapsed() - start_offset);

    start_offset = now.elapsed();
    let mut sign_buffer = [false; 64];

    for n in 0..NUM_POINTS / 64 {
        buffer.try_fill(&mut rng).unwrap();
        sign_buffer.try_fill(&mut rng).unwrap();

        let mask = Mask::from_array(sign_buffer);
        polar_points_y[n] = f32x64::from_array(buffer).ln() / simd_height_lambda;
        polar_points_y[n] = mask.select(-polar_points_y[n], polar_points_y[n]).sin();
    }
    println!("y sample took {:?}", now.elapsed() - start_offset);

    start_offset = now.elapsed();

    for n in 0..NUM_POINTS / 64 {
        buffer.try_fill(&mut rng).unwrap();
        polar_points_z[n] = f32x64::from_array(buffer);
    }

    println!("z sample took {:?}", now.elapsed() - start_offset);
    start_offset = now.elapsed();

    let mut cartesian_buffer = [0.; 4];
    let mut cartesian = vec![f32x4::splat(0.); NUM_POINTS];
    for outer in 0..NUM_POINTS / 64 {
        for inner in 0..64 {
            cartesian_buffer[0] = polar_points_x[outer][inner];
            cartesian_buffer[1] = polar_points_y[outer][inner];
            cartesian_buffer[2] = polar_points_z[outer][inner];
            cartesian_buffer[3] = 0.0;
            if cartesian_buffer[0].is_nan() {
                cartesian_buffer[0] = 0.0;
            }
            if cartesian_buffer[1].is_nan() {
                cartesian_buffer[1] = 0.0;
            }
            if cartesian_buffer[2].is_nan() {
                cartesian_buffer[2] = 0.0;
            }
            cartesian[outer * 64 + inner] = f32x4::from_array(cartesian_buffer);
        }
    }

    println!(
        "cartesian conversion took {:?}",
        now.elapsed() - start_offset
    );
    let done = now.elapsed();
    trick_optimization(&cartesian);

    done
}

fn trick_optimization(cartesian: &Vec<f32x4>) {
    // for n in 0..NUM_POINTS {
    //     println!("{:?}", cartesian[n]);
    // }

    let mut sum: f32 = 0.;
    for n in 0..NUM_POINTS {
        let pre_sum = sum
            + (cartesian[n][0] + cartesian[n][1] + cartesian[n][2])
                .to_f32()
                .unwrap();
        if pre_sum.is_nan() {
            println!(
                "nan for n: {}, x: {}, y: {}, z: {}",
                n, cartesian[n][0], cartesian[n][1], cartesian[n][2]
            );
            panic!("sum: {}", sum);
        }
        sum += (cartesian[n][0] + cartesian[n][1] + cartesian[n][2])
            .to_f32()
            .unwrap();
    }
    println!("sum: {}", sum);
    // print all elements
}
