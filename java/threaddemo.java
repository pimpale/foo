import java.util.concurrent.atomic.*;


public class threaddemo {
		public static void main(String[] args)
		{
				int[] a = {0};
				int[] b = {0};
				Thread thread = new Thread(new MyThread(a,b, 5));
				thread.start();
				System.out.println("slow");
		}
}


/**
 * Takes two arrays and adds them
 *
 *
 */
class MyThread implements Runnable {
		
		private AtomicInteger lock = new AtomicInteger(0);
		private int[] a;
		private int[] b;
		private int[] c;
		private int id;

		public MyThread(int[] a, int[] b, int id)
		{
				this.a = a;
				this.b = b;
				this.id = id;
		}


		@Override
		public void run()
		{
				lock.decrementAndGet();
				c = new int[a.length];
				for(int i = 0; i < a.length; i++)
				{
						c[i] = a[i] + b[i];
				}
				lock.incrementAndGet();
		}
}
