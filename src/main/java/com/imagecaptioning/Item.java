/**
 * 
 */
package com.imagecaptioning;

/**
 * @author Arnaud
 *
 */

public class Item {

	private String imageFileName;
	private String imageLabel;
	private String path;
	
    public Item() {
		
	}
	
	public Item(String imageFileName, String imageLabel) {
		this.imageFileName = imageFileName;
		this.imageLabel = imageLabel;
		//this.path = path;
	}
	
	// Getters and Setters
	
	public String getImageFileName() {
		return imageFileName;
	}
		
	public void setImageFileName(String imFn) {
		this.imageFileName = imFn;
	}
		
	public String getImageLabel() {
		return imageLabel;
	}
	
	public void setPath(String aPath) {
		this.path = aPath;
	}
	
	public String getPath() {
		return path;
	}
}
