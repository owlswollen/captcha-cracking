import java.io.IOException;
import java.io.File;
import java.util.Scanner;

import org.jsoup.Connection;
import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import java.awt.Color;
import java.awt.Image;
import java.awt.image.BufferedImage;
import javax.swing.ImageIcon;
import javax.imageio.ImageIO;
import java.awt.Graphics2D;
import javax.swing.JFrame;
import javax.swing.JTextField;
import javax.swing.JPasswordField;
import javax.swing.JOptionPane;
import java.io.ByteArrayInputStream;
import java.awt.image.DataBufferInt;
/**
 * Hello world!
 *
 */
public class App
{
    public static void main( String[] args )
    {
	String dir_name = System.getProperty("user.dir")+"/Downloads";
	//System.out.println(dir_name);
	File theDir = new File(dir_name);
	if (!theDir.exists()){
	    theDir.mkdirs();
	}

	Integer count = 500;

	System.out.println( "CAPTCHA Test Generation" );
	String url ="https://en.wikipedia.org/w/index.php?title=Special:CreateAccount&returnto=Main+Page";
	Document document;
	Connection conn;

	try {
	    for (int i = 1; i <= count; i++) {
		String file_name = dir_name + "/img_"+i+".png";
		conn = Jsoup.connect(url);
		document = conn.get();
		Element capt = document.select ("img").first();
		Connection.Response response;
		response = Jsoup.connect(capt.absUrl("src")).cookies(conn.response().cookies()).ignoreContentType(true).execute();
		ImageIcon image = new ImageIcon(ImageIO.read(new ByteArrayInputStream(response.bodyAsBytes())));

		// This is to output onto the window
		//JOptionPane.showMessageDialog(null, image, "Captcha image", JOptionPane.PLAIN_MESSAGE);

		// This is to save the image into the directory
		Image img = image.getImage();
		BufferedImage bi = new BufferedImage(img.getWidth(null),img.getHeight(null),BufferedImage.TYPE_4BYTE_ABGR);
		Graphics2D g2 = bi.createGraphics();
		g2.drawImage(img, 0, 0, null);
		g2.dispose();
		ImageIO.write(bi, "png", new File(file_name));
		System.out.println("["+i+"/"+count+"] Generating test cases...");

	    }
	} catch (IOException e) {
	    // TODO Auto-generated catch block
	    e.printStackTrace();
	}
    }

}
