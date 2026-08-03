#[derive(Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f32>,
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn get(&self, row: usize, col: usize) -> f32 {
        let index = row * self.cols + col;
        self.data[index]
    }

    pub fn set(&mut self, row: usize, col: usize, val: f32) {
        let index = row * self.cols + col;
        self.data[index] = val;
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f32>) -> Matrix {
        assert_eq!(data.len(), rows * cols);
        Matrix { rows, cols, data }
    }

    pub fn display(&self) {
        for row in 0..self.rows {
            for col in 0..self.cols {
                print!("{:.4} ", self.get(row, col));
            }
            println!();
        }
        println!();
    }
}

#[allow(dead_code)]
fn main() {
    println!("matrix crate main");
}
