//! Minimal columnar CSV reader for the subhalo-finder catalogs (header-addressed,
//! tolerant of column order and extra columns).

use std::collections::HashMap;
use std::path::Path;

pub struct Table {
    idx: HashMap<String, usize>,
    num: Vec<Vec<f64>>,
    text: HashMap<usize, Vec<String>>,
    pub len: usize,
}

impl Table {
    pub fn read(path: &Path) -> Result<Table, String> {
        let s = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Table::parse(&s).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn parse(s: &str) -> Result<Table, String> {
        let mut lines = s.lines().filter(|l| !l.trim().is_empty());
        let header = lines.next().ok_or("empty csv")?;
        let names: Vec<String> = header.split(',').map(|c| c.trim().to_ascii_lowercase()).collect();
        let nc = names.len();
        let idx: HashMap<String, usize> = names.iter().enumerate().map(|(i, n)| (n.clone(), i)).collect();
        let mut num: Vec<Vec<f64>> = vec![Vec::new(); nc];
        let mut text: HashMap<usize, Vec<String>> = HashMap::new();
        let mut is_text = vec![None::<bool>; nc];
        let mut len = 0;
        for line in lines {
            let f: Vec<&str> = line.split(',').map(|x| x.trim()).collect();
            for c in 0..nc {
                let cell = f.get(c).copied().unwrap_or("");
                let parsed = cell.parse::<f64>();
                if is_text[c].is_none() {
                    is_text[c] = Some(parsed.is_err() && !cell.is_empty() && !cell.eq_ignore_ascii_case("nan"));
                }
                num[c].push(parsed.unwrap_or(f64::NAN));
                if is_text[c] == Some(true) {
                    text.entry(c).or_default().push(cell.to_string());
                }
            }
            len += 1;
        }
        Ok(Table { idx, num, text, len })
    }

    pub fn f(&self, name: &str) -> Result<&[f64], String> {
        let c = *self.idx.get(name).ok_or(format!("missing column '{name}'"))?;
        Ok(&self.num[c])
    }

    pub fn s(&self, name: &str) -> Result<Vec<String>, String> {
        let c = *self.idx.get(name).ok_or(format!("missing column '{name}'"))?;
        Ok(match self.text.get(&c) {
            Some(v) => v.clone(),
            None => self.num[c].iter().map(|x| x.to_string()).collect(),
        })
    }
}
