import java.io.*;
import java.awt.image.*;
import javax.imageio.*;

class loadbufferedimage {

  public static void main(String[] args) throws Throwable {
    BufferedImage image = ImageIO.read(new File("test.png"));
  }
}
