use std::time::{Duration, Instant};

use rand::{rngs::ThreadRng, thread_rng};
use rand_distr::{Distribution, Exp};

use crate::{GALAXY_HEIGHT, GALAXY_RADIUS, NUM_POINTS};

// per run 2.8 s for 100_000_000 points
#[allow(dead_code)]
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
