#![feature(proc_macro_hygiene)]
#![feature(register_tool)]
#![register_tool(furiosa_opt)]
use furiosa_opt_std::prelude::*;
axes![A = 2048];
type Chip = m![1];
#[device(chip = 1)]
pub fn identity(_device: &mut Device, x: &HbmTensor<f32, Chip, m![A]>) -> HbmTensor<f32, Chip, m![A]> { x.clone() }
