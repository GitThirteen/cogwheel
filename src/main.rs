mod ml;

use std::fs;
use image::{imageops, GrayImage, ImageReader};
use ml::Trainable;

fn fetch_data_from(path: &str, dim: u32) -> (Vec<f32>, Vec<Vec<f32>>) {
    let mut labels: Vec<f32> = Vec::<f32>::new();
    let mut data = Vec::<Vec<f32>>::new();

    let directory = fs::read_dir(path).unwrap();

    for file in directory {
        let path = file.unwrap().path();
        let name = path.file_name().unwrap().to_str().unwrap();

        let label = if name.starts_with("0") { 1.0 } else { -1.0 };
        labels.push(label);

        let image = ImageReader::open(&path).unwrap().decode().unwrap().to_luma32f();
        let mut resized = imageops::resize(&image, dim, dim, imageops::FilterType::Lanczos3);
        for px in resized.pixels_mut() {
            px.0[0] /= 255.0;
        }
        data.push(resized.into_vec());
    }

    (labels, data)
}

fn main() {
    let learning_rate = 0.2;
    let dim = 50;
    let training_data_path = "./training_data";
    let testing_data_path = "./testing_data";

    let (learning_labels, learning_data) = fetch_data_from(&training_data_path, dim);
    let (testing_labels, testing_data) = fetch_data_from(&testing_data_path, dim);
    
    let mut perceptron = ml::Perceptron::new(learning_rate, 100);
    let misses = perceptron.fit(learning_data, learning_labels);

    let preds = perceptron.predict(testing_data);
    let mut fails = 0;
    for (i, label) in testing_labels.iter().enumerate() {
        if *label != preds[i] {
            fails += 1;
        }
    }

    println!("{}", "Success rate: ".to_owned() + &(100.0 * (1.0 - (fails as f32 / testing_labels.len() as f32))).to_string() + "%");

    println!("Misses after...");
    for (i, miss) in misses.iter().enumerate() {
        print!("{}", "[".to_owned() + &(i + 1).to_string() + "] = " + &miss.to_string() + " | ");
    }

    let mut buffer: GrayImage = image::ImageBuffer::new(dim, dim);
    for (x, y, px) in buffer.enumerate_pixels_mut() {
        let i = y * dim + x;
        px.0[0] = (255.0 * 255.0 * perceptron.weights[i as usize]) as u8;
    }

    buffer.save("weights.jpg").unwrap();
}