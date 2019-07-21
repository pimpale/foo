import java.util.Scanner;

class Main {
		
		static int[] getArray()
		{
				int[] arr = new int[10];
				Scanner scanner = new Scanner(System.in);
				int currentNum = 0;
				int currentIndex = 0;
				while(currentNum >= 0 && currentIndex < 10)
				{
						currentNum = scanner.nextInt();
						currentIndex++;
						arr[currentIndex] = currentNum;
				}
				return arr;
		}


		static int[] mergeArray(int[] arr1, int[] arr2)
		{


		public static void main(String[] args)
		{
				int[] arr1 = getArray();
				int[] arr2 = getArray();



				for(int i = 0; i < 10; i++)
				{
						System.out.println(arr[i]);
				}
		}

}
