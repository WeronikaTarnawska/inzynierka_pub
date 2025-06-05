/* ==========================================================================================
 *                          This file is part of the Bachelor Thesis project
 *                                   University of Wrocław
 *                         Author: Weronika Tarnawska (Index No. 331171)
 *                                         June 2025
 * ========================================================================================== */
pub mod wav;
pub mod utils;
pub mod signal;
pub mod fft;
pub mod lms;
pub mod wiener;
pub mod ss;

#[cfg(test)]
const _EPSILON: f64 = 1e-12;
