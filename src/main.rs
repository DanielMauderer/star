#![feature(portable_simd)]

const GALAXY_RADIUS: f32 = 1. / 30003.2615637769;
const GALAXY_HEIGHT: f32 = 1. / 2503.2615637769;
const NUM_POINTS: usize = 100_000_000;

fn main() {
    println!("go");
    let mut done = simd::run();

    println!("processing time: {:?}", done);
    done = unoptimized::run();
    println!("processing time: {:?}", done);
}

mod simd {
    use std::{
        simd::{f32x4, f32x64, Mask, StdFloat},
        time::{Duration, Instant},
    };

    use rand::{thread_rng, Fill};
    use rand_distr::num_traits::ToPrimitive;

    use crate::{GALAXY_HEIGHT, GALAXY_RADIUS, NUM_POINTS};

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
}

mod unoptimized {
    use std::time::{Duration, Instant};

    use rand::{rngs::ThreadRng, thread_rng};
    use rand_distr::{Distribution, Exp};

    use crate::{GALAXY_HEIGHT, GALAXY_RADIUS, NUM_POINTS};

    // per run 2.8 s for 100_000_000 points
    pub fn run() -> Duration {
        let now = Instant::now();
        let mut polar_points = vec![(0., 0., 0.); NUM_POINTS];

        let exp_radius = Exp::new(GALAXY_RADIUS).unwrap();
        let exp_height = Exp::new(GALAXY_HEIGHT).unwrap();

        let mut rng = thread_rng();

        for n in 0..NUM_POINTS {
            polar_points[n].0 = exp_radius.sample(&mut rng);
            polar_points[n].1 = get_point_height(&exp_height, &mut rng);
            polar_points[n].2 = rand::random::<f32>();
        }
        let done = now.elapsed();

        trick_optimization(&mut polar_points);
        done
    }

    fn trick_optimization(polar_points: &mut Vec<(f32, f32, f32)>) {
        let mut sum = 0.0;
        for n in 0..NUM_POINTS {
            sum += polar_points[n].0 + polar_points[n].1 + polar_points[n].2;
        }
        println!("sum: {}", sum);
    }

    fn get_point_height(distribution: &Exp<f32>, rng: &mut ThreadRng) -> f32 {
        match rand::random() {
            true => distribution.sample(rng),
            false => -distribution.sample(rng),
        }
    }
}

// mod simd_rand {
//     use std::simd::{f32x64, num::SimdFloat, u64x64, Simd};
//
//     use rand::{thread_rng, Fill};
//
//     const a: Simd<u64, 64> = u64x64::from_array([48271; 64]);
//     const m: Simd<u64, 64> = u64x64::from_array([2147483647; 64]);
//     const mf: f32x64 = f32x64::from_array([2147483647.0; 64]);
//     static mut next: u64x64 = u64x64::from_array([0; 64]);
//
//     pub fn srand() {
//         let mut seed = [0; 64];
//         let mut rng = thread_rng();
//
//         (&mut seed).try_fill(&mut rng).unwrap();
//
//         unsafe { next = u64x64::from_array(seed) };
//     }
//     //generate a random number between 0 and 1 with Lehmer random number generator
//     #[inline]
//     pub fn sample() -> f32x64 {
//         unsafe { f32x64::from_bits(a * next % m) / mf }
//     }
// }
