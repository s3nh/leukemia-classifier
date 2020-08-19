use tract_onnx::prelude::*;
use image::*;
use image::imageops::*;


// read facedetector.onnx and check if it does not fuck up with  
// onnx version

fn main() -> TractResult<()> {
    let model = tract_onnx::onnx()
        .model_path("../model/FaceDetector.onnx")?
        .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, 224, 224)))?
        .into_optimized()?
        .into_runnable()?;
    Ok(());
    // open image and check if is it worth something 
    
    let image = image::open("test.jpg").unwrap().to_rgb();
    let resized = image::imageops::resize(&image, 224, 224, ::image::imageops::FilterType::Triangle);
    let image: Tensor = tract_ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)|{
        resized[(x as _, y as _)][c] as f32 
    })
    .into();

    let result = model.run(tvec!(image))?;
    println!("{}", result);
}
