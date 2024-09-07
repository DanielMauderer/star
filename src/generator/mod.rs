pub mod simd {

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
}

pub mod unoptimized {
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
}

pub mod gpu {
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
        let file = std::fs::File::create("points.txt")?;
        let mut writer = std::io::BufWriter::new(file);
        let mut start = duration.elapsed();
        start = duration.elapsed() - start;
        fill_points(&pro_que, &mut points, &duration)?;

        for i in 0..10 {
            println!(
                "t: {:?}/ i: {:?} Iteration: {} - Done filling points...",
                duration.elapsed(),
                start,
                i
            );

            let write_job = write_batch_to_file(&mut writer, points.clone());

            println!("{:?} - Done writing results...", duration.elapsed());
            fill_points(&pro_que, &mut points, &duration)?;
            write_job.await?;
            println!(
                "t: {:?}/ i: {:?} Iteration: {} - Done writing points...",
                duration.elapsed(),
                start,
                i
            );
        }

        println!("{:?} - all done", duration.elapsed());
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
}
