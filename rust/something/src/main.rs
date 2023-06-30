use tokio::sync::mpsc;

#[tokio::main]
async fn main() {
    let (tx, mut rx) = broadcast::channel::<()>(16);

    tokio::spawn(async move {
        // sleep 100 ms
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        // try to grab as many as possible
        while let Ok(()) = rx.recv().await {
            println!("Hello, world!");
        }
        println!("ended!");
    });

    tx.send(()).unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    tx.send(()).unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    tx.send(()).unwrap();
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
    tx.send(()).unwrap();
    drop(tx);
    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
}
