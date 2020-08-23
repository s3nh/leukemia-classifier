use actix_web::Error;
use actix_web::{web, web::Path, App, HttpRequest, HttpResponse, HttpServer, Responder};
use futures::{
    future::{ok, Future}, 
    Stream, 
};

use rand::prelude::*;
use std::fs::{File, OpenOptions};
use std::io::Write;

fn flush_stdout() {
    std::io::stdout().flush().unwrap();
}

fn delete_file(info : Path<(String, )>) -> impl Responder{
    let filename = &info.0;
    flush_stdout();
    
    match std::fs::remove_file(&filename) {
        Ok(_) => {
            HttpResponse::Ok()
        }
        Err(error) => {
            HttpResponse::NotFound()
        }
    }
}

fn download_file(info: Path<(String,)>) -> impl Responder {
    let filename = &info.0;
    flush_stdout();
    
    fn read_file_contents(filename: &str) -> std::io::Result<String> {
        use std::io::Read;
        let mut contents = String::new();
        File::open(filename)?.read_to_string(&mut contents)?;
        Ok(contents)
    }

    match read_file_contents(&filename) {
        Ok(contents) => {
            println!("Downloaded file \"{}\"", filename);
            HttpResponse::Ok().content_type("text/plain").body(contents)
        }

        Err(error) => {
            println!("Failed to read file \"{}\": {}", filename, error);
            HttpResponse::NotFound().finish()
        }
    }
}

fn upload_specified_file(
    payload: web::Payload, 
    info: Path<(String,)>,) -> impl Future<Item = HttpResponse, Error = Error> {
    let filename = info.0.clone();
    print!("Uploading file \"{}*.txt\", ... ", filename);
    flush_stdout();

    payload
        .map_err(Error::from)
        .fold(web::BytesMut::new(), move |mut body, chunk| {
            body.extend_from_slice(&chunk);
            Ok::<_, Error>(body)
        })
        .and_then(move |contents| {
            let f = File::create(&filename);
            if f.is_err() {
                println!("Failed to crate file \"{}\"", filename);
                return ok(HttpResponse::NotFound().into());
        }
        
        if f.unwrap().write_all(&contents).is_err() {
            println!("Failed to write file \"{}\"", filename);
            return ok(HttpResponse::NotFound().into());
        }
        
        println!("Uploaded file \"{}\"", filename);
        ok(HttpResponse::Ok().finish())
    })
}


// Upload a new file

fn upload_new_file(
    payload: web::Payload, 
    info: Path<(String,)>, ) -> impl Future<Item = HttpResponse, Error = Error> { 
    // assign filename_prefix 
    let filename_prefix = info.0.clone();
    println!("Uploading file \"{}*.txt\" ... ", filename_prefix);
    flush_stdout();

    payload
        .map_err(Error::from)
        .fold(web::BytesMut::new(), move |mut body, chunk| {
            body.extend_from_slice(&chunk);
            Ok::<_, Error>(body)
        })
        .and_then(move |contents| {
        // rng 
            let mut rng = rand::thread_rng();
            let mut attempts = 0;
            let mut file;
            let mut filename;
            const MAX_ATTEMPTS: u32 = 100;

            loop {
                attempts += 1;
                if attempts > MAX_ATTEMPTS { 
                    println!(
                        "Failes to create new file with prefix \"{}\", \
                        after {} attempts.", 
                        filename_prefix, MAX_ATTEMPTS
                );
                return ok(HttpResponse::NotFound().into());
            }

            //Generate 3-digit pseudo random number
            //ill use it to create a filename 
           
            filename = format!("{}{:03}.txt", filename_prefix, rng.gen_range(0, 1000));
            
            // Create not existing file 
            file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&filename);

            // Exit the loop if file is properly created 
            if file.is_ok() {
                break;
                }
        }

        // Write contents into file asynchronously 
        if file.unwrap().write_all(&contents).is_err() {
            println!("Failed to write file \"{}\"", filename);
            return ok(HttpResponse::NotFound().into());
        }
        
        println!("Uploaded file \"{}\"", filename);
        ok(HttpResponse::Ok().content_type("text/plain").body(filename))
        })
}

fn invalid_resource(req: HttpRequest) -> impl Responder {
    println!("Invalid URI: \"{}\"", req.uri());
    HttpResponse::NotFound()
}


fn main() -> std::io::Result<()> {
    let server_address = "localhost:8080";
    println!("Listening at address {} ...", server_address);
    HttpServer::new(|| {
        App::new()
            .service(
                web::resource("/{filename}")
                    .route(web::delete().to(delete_file))
                    .route(web::get().to(download_file))
                    .route(web::put().to_async(upload_specified_file))
                    .route(web::post().to_async(upload_new_file)),
                )
                .default_service(web::route().to(invalid_resource))
            })
            .bind(server_address)?
            .run()
        }




