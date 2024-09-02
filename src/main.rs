use std::time::{Duration, Instant};

use rand::{rngs::ThreadRng, thread_rng};
use rand_distr::{Distribution, Exp};

// pro optimiyation 2.8 s for 100_000_000 points

const GALAXY_RADIUS: f64 = 1. / 30003.2615637769;
const GALAXY_HEIGHT: f64 = 1. / 2503.2615637769;
const NUM_POINTS: usize = 100000000;

fn main() {
    println!("go");

    let done = unoptimized();
    // unoptimized(&mut polar_points);

    println!("processing time: {:?}", done);
}

fn get_point_height(distribution: &Exp<f64>, rng: &mut ThreadRng) -> f64 {
    match rand::random() {
        true => distribution.sample(rng),
        false => -distribution.sample(rng),
    }
}

fn unoptimized() -> Duration {
    let now = Instant::now();
    let mut polar_points = vec![(0., 0., 0.); NUM_POINTS];

    let exp_radius = Exp::new(GALAXY_RADIUS).unwrap();
    let exp_height = Exp::new(GALAXY_HEIGHT).unwrap();

    let mut rng = thread_rng();

    for n in 0..NUM_POINTS {
        polar_points[n].0 = exp_radius.sample(&mut rng);
        polar_points[n].1 = get_point_height(&exp_height, &mut rng);
        polar_points[n].2 = rand::random::<f64>();
    }
    let done = now.elapsed();

    trick_optimization(&mut polar_points);
    done
}

fn trick_optimization(polar_points: &mut Vec<(f64, f64, f64)>) {
    let mut sum = 0.0;
    for n in 0..NUM_POINTS {
        sum += polar_points[n].0 + polar_points[n].1 + polar_points[n].2;
    }
    println!("sum: {}", sum);
}
