use std::convert::Infallible;
use std::net::SocketAddr;
use hyper::{Body, Request, Response, Server};
use hyper::service::{make_service_fn, service_fn};

async fn hello_world(_req: Request<Body>) -> Result<Response<Body>, Infallible> {
    Ok(Response::new("Hello, World".into()))
}

// By default, tokio spawns as many threads as there are CPU cores. This is undesirable,
// because you need to specify in the Gramine manifest the maximal number of threads per
// process, and ideally this wouldn't depend on your hardware.
//
// See sgx.thread_num in the manifest.
#[tokio::main(worker_threads = 4)]
async fn main() {
    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));

    let make_service = make_service_fn(|_conn| async {
        Ok::<_, Infallible>(service_fn(hello_world))
    });

    let server = Server::bind(&addr).serve(make_service);

    if let Err(e) = server.await {
        eprintln!("server error: {}", e);
        std::process::exit(1);
    }
}
