mod ml;

use std::fs;
use image::{imageops, GrayImage, ImageReader};
use plotters::prelude::*;
use ml::Trainable;

fn fetch_data_from(path: &str, dim: u32) -> (Vec<f32>, Vec<Vec<f32>>) {
    let mut labels: Vec<f32> = Vec::<f32>::new();
    let mut data = Vec::<Vec<f32>>::new();

    let directory = fs::read_dir(path).unwrap();

    for file in directory {
        let path = file.unwrap().path();
        let name = path.file_name().unwrap().to_str().unwrap();

        let label = if name.starts_with("0") { -1.0 } else { 1.0 };
        labels.push(label);

        let image = ImageReader::open(&path).unwrap().decode().unwrap().to_luma32f();
        let mut resized = imageops::resize(&image, dim, dim, imageops::FilterType::Lanczos3);
        
        let is_full_range = resized
            .iter()
            .any(|&v| v > 1.0 + f32::EPSILON);

        if is_full_range {
            for px in resized.pixels_mut() {
                px.0[0] /= 255.0;
            }
        }

        data.push(resized.into_vec());
    }

    (labels, data)
}

fn main() {
    let learning_rate = 0.25;
    let dim = 16;
    let epochs = 1000;
    let training_data_path = "./training_data";
    let testing_data_path = "./training_data";

    let (learning_labels, learning_data) = fetch_data_from(&training_data_path, dim);
    let (testing_labels, testing_data) = fetch_data_from(&testing_data_path, dim);
    
    let mut perceptron = ml::Perceptron::new(learning_rate, epochs);
    let misses = perceptron.fit(learning_data, learning_labels);
    let preds = perceptron.predict(testing_data);

    let fails = testing_labels
        .iter()
        .zip(&preds)
        .filter(|&(a, b)| a != b)
        .count();

    let root = BitMapBackend::new("out.png", (1024, 768)).into_drawing_area();
    root.fill(&WHITE);

    let chart = ChartBuilder::on(&root)
        .margin(10)
        .caption("Test", ("sans-serif", 40))
        .build_cartesian_2d(1..epochs, 0..testing_labels.len());

    chart.unwrap().draw_series(
        LineSeries::new(
            misses
            .iter()
            .enumerate()
            .map(|(i, miss_count)| (i, *miss_count)), &BLUE)
    ).unwrap();

    root.present().unwrap();

    println!("{}", "Success rate: ".to_owned() + &(100.0 * (1.0 - (fails as f32 / testing_labels.len() as f32))).to_string() + "%");

    let mut buffer: GrayImage = image::ImageBuffer::new(dim, dim);
    for (x, y, px) in buffer.enumerate_pixels_mut() {
        let i = y * dim + x;
        px.0[0] = (255.0 * perceptron.weights[i as usize]) as u8;
    }

    buffer.save("weights.jpg").unwrap();
}