use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::Path;

/// A single epochs recorded metrics
#[derive(Debug, Clone)]
pub struct EpochRecord {
    pub epoch: usize,
    pub loss: f32,
    pub train_accuracy: Option<f32>,
    pub elapsed_s: f32,
    pub extra: Vec<(String, f32)>,
}

/// Accumulator and zero-dependency JSON exporter for model training metrics
#[derive(Debug, Clone)]
pub struct MetricsLogger {
    pub model_id: String,
    pub model_name: String,
    pub dataset: String,
    pub task: String,
    pub architecture: String,
    pub optimiser: String,
    pub initial_loss: Option<f32>,
    pub test_accuracy: Option<f32>,
    pub epochs: Vec<EpochRecord>,
}

impl MetricsLogger {
    /// Creates a new metrics logger with the given model metadata
    pub fn new(
        model_id: impl Into<String>,
        model_name: impl Into<String>,
        dataset: impl Into<String>,
        task: impl Into<String>,
        architecture: impl Into<String>,
        optimiser: impl Into<String>,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            model_name: model_name.into(),
            dataset: dataset.into(),
            task: task.into(),
            architecture: architecture.into(),
            optimiser: optimiser.into(),
            initial_loss: None,
            test_accuracy: None,
            epochs: Vec::new(),
        }
    }

    /// Sets the intial un-trained loss
    pub fn set_init_loss(&mut self, loss: f32) {
        self.initial_loss = Some(loss);
    }

    /// Sets the final evaluated test accuracy
    pub fn set_test_accuracy(&mut self, test_acc: f32) {
        self.test_accuracy = Some(test_acc);
    }

    /// Records the metrics for a single epoch
    pub fn log_epoch(
        &mut self,
        epoch: usize,
        loss: f32,
        train_accuracy: Option<f32>,
        elapsed_s: f32,
    ) {
        self.epochs.push(EpochRecord {
            epoch,
            loss,
            train_accuracy,
            elapsed_s,
            extra: Vec::new(),
        });
    }

    /// Records metrics for a single epoch with additional domain-specific metrics (e.g. recon_loss, kl_loss, beta)
    pub fn log_epoch_with_extra(
        &mut self,
        epoch: usize,
        loss: f32,
        train_accuracy: Option<f32>,
        elapsed_s: f32,
        extra: Vec<(&str, f32)>,
    ) {
        self.epochs.push(EpochRecord {
            epoch,
            loss,
            train_accuracy,
            elapsed_s,
            extra: extra.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        });
    }

    /// Generates a cleanly formatted JSON string representing the logged metrics
    pub fn to_json(&self) -> String {
        let mut json = String::new();
        json.push_str("{\n");
        json.push_str(&format!("  \"id\": \"{}\",\n", escape_json(&self.model_id)));
        json.push_str(&format!(
            "  \"name\": \"{}\",\n",
            escape_json(&self.model_name)
        ));
        json.push_str(&format!(
            "  \"dataset\": \"{}\",\n",
            escape_json(&self.dataset)
        ));
        json.push_str(&format!("  \"task\": \"{}\",\n", escape_json(&self.task)));
        json.push_str(&format!(
            "  \"architecture\": \"{}\",\n",
            escape_json(&self.architecture)
        ));
        json.push_str(&format!(
            "  \"optimiser\": \"{}\",\n",
            escape_json(&self.optimiser)
        ));
        if let Some(init_loss) = self.initial_loss {
            json.push_str(&format!("  \"initial_loss\": {:.4},\n", init_loss));
        } else {
            json.push_str("  \"initial_loss\": null,\n");
        }
        if let Some(test_acc) = self.test_accuracy {
            json.push_str(&format!("  \"test_accuracy\": {:.2},\n", test_acc));
        } else {
            json.push_str("  \"test_accuracy\": null,\n");
        }

        json.push_str("  \"epochs\": [\n");
        for (i, epoch) in self.epochs.iter().enumerate() {
            json.push_str("    {\n");
            json.push_str(&format!("      \"epoch\": {},\n", epoch.epoch));
            json.push_str(&format!("      \"loss\": {:.4},\n", epoch.loss));
            if let Some(acc) = epoch.train_accuracy {
                json.push_str(&format!("      \"train_accuracy\": {:.2},\n", acc));
            } else {
                json.push_str("      \"train_accuracy\": null,\n");
            }
            json.push_str(&format!("      \"elapsed_s\": {:.2}", epoch.elapsed_s));
            if !epoch.extra.is_empty() {
                json.push_str(",\n");
                for (j, (k, v)) in epoch.extra.iter().enumerate() {
                    json.push_str(&format!("      \"{}\": {:.4}", escape_json(k), v));
                    if j + 1 < epoch.extra.len() {
                        json.push_str(",\n");
                    } else {
                        json.push('\n');
                    }
                }
            } else {
                json.push('\n');
            }
            if i + 1 < self.epochs.len() {
                json.push_str("    },\n");
            } else {
                json.push_str("    }\n");
            }
        }
        json.push_str("  ]\n");
        json.push_str("}\n");

        json
    }

    /// Saves the logged metrics as a JSON file at `path`, creating parent directories if needed
    pub fn save_to_json<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                create_dir_all(parent)?;
            }
        }
        let mut file = File::create(path)?;
        file.write_all(self.to_json().as_bytes())?;
        Ok(())
    }
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}
