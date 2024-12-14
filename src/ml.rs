use rand::Rng;

pub trait Trainable {
    fn fit(&mut self, data: Vec<Vec<f32>>, labels: Vec<f32>) -> Vec<u32>;
    fn forward(&self, data: Vec<Vec<f32>>) -> Vec<f32>;
    fn predict(&self, data: Vec<Vec<f32>>) -> Vec<f32>;
}

pub struct Perceptron {
    learning_rate: f32,
    epochs: u32,
    pub weights: Vec<f32>,
    bias: f32
}

impl Perceptron {
    pub fn new(learning_rate: f32, epochs: u32) -> Self {
        Self { learning_rate, epochs, weights: vec![], bias: 0.0 }
    }
}

impl Trainable for Perceptron {
    fn fit(&mut self, data: Vec<Vec<f32>>, labels: Vec<f32>) -> Vec<u32> {
        assert!(data.len() > 0);

        let n_labels = data.len();
        let n_data = data[0].len();

        self.weights = vec![0.0; n_data];
        let mut misses_per_epoch = vec![];

        for _ in 0..self.epochs {
            let n = rand::thread_rng().gen_range(0..(n_labels - 1));

            let actual = labels[n];
            let fetched = self.forward(vec![data[n].clone(); 1]);
            let error = actual - fetched[0];

            self.bias += self.learning_rate * error;
            for (i, weight) in self.weights.iter_mut().enumerate() {
                *weight += self.learning_rate * error * data[n][i];
            }

            let n_correct = self.forward(data.clone())
                .iter()
                .zip(&labels)
                .map(|(pred, actual)| if (actual - pred) == 0.0 { 1 } else { 0 })
                .sum::<u32>();

            misses_per_epoch.push(n_labels as u32 - n_correct);
        }

        misses_per_epoch
    }

    fn forward(&self, data: Vec<Vec<f32>>) -> Vec<f32> {
        assert!(data.len() > 0);
        assert_eq!(self.weights.len(), data[0].len());

        let mut predictions = vec![];
        for vector in data {
            let mut w_sum = 0.0;
            for (i, value) in vector.iter().enumerate() {
                w_sum += value * self.weights[i];
            }
            w_sum += self.bias;
            predictions.push(if w_sum < 0.0 { -1.0 } else { 1.0 });
        }

        predictions
    }

    fn predict(&self, data: Vec<Vec<f32>>) -> Vec<f32> {
        self.forward(data)
    }
}