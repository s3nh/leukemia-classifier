use ndarray::Array;
use tract_onnx::prelude::*;

// read facedetector.onnx and check if it does not fuck up with  
// onnx version

fn main() -> Result<()>{
    let mut  model = tract_onnx::onnx()
        .model_for_path("../model/FaceDetector.onnx")?;
    /*

        model.set_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, 224, 224)))?;
        let model = model.into_optimized()?;
        println("{}", model);
        let plan = SimplePlan::new(model)?;
    // open image and check if is it worth something 
    
    let image = image::open("test.jpg").unwrap().to_rgb();
    let resized = image::imageops::resize(&image, 224, 224, ::image::imageops::FilterType::Triangle);
    let image: Tensor = tract_ndarray::Array4::from_shape_fn((1, 3, 224, 224), |(_, c, y, x)|{
        resized[(x as _, y as _)][c] as f32 
    })
    .into();

    let result = plan.run(tvec!(image))?;
    println!("{}", result);
    */
    Ok(());
    println("{}", model);
}
