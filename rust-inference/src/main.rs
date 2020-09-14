use ndarray::Array;
use tract_onnx::prelude::*;
// Examples were updated to higher version of rustupdate so 
//  lets assume that we provide path into string like way
//
//
fn main() -> TractResult<()>{
    let mut  model = tract_onnx::onnx()
        .model_for_path("../classification/leukemia_resnet50.onnx")?
        .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 3, 450, 450)))?
        .into_optimized()?
        .into_runnable()?;
    
    let image  = image::open("test.bmp").unwrap().to_rgb();
    let resized = image::imageops::resize(&image, 40, 450, ::image::imageops::FilterType::Triangle);
    let image: Tensor = tract_ndarray::Array4::from_shape_fn((1, 3, 450, 450), |(_,c,y,x)| {
        resized[(x as _, y as _)][c] as f32/255.0
    })
    .into();

    let result = model.run(tvec!(image))?;
    let best = result[0]
        .to_array_view::<f32>()?
        .iter()
        .cloned()
        .zip(1..)
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    match best {
        Some(t) => Ok(t),
        None => Ok((0.0, 0)),
    }    
}
