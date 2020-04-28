/**
 * 
 */
package com.imagecaptioning;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * @author Arnaud
 *
 */

public class Utils {
	
	private static final Logger log = LoggerFactory.getLogger(Utils.class);

	/**
     *   Helper method that takes in a string representing the full path of the csv file containing the train/test data
     *   @param filePath.
     */
    public List<Item> readCsvDataFile(String filePath) {
    	
    	// Create a new list of line instances to be fill
		List<Item> itemList = new ArrayList<Item>();
    	BufferedReader fileReader = null;
    	
    	try {   		
    		String line = "";
    		
    		// Create the fileReader
    		fileReader = new BufferedReader(new FileReader(filePath));
    		
    		// Read the CSV file header to skip it
    		//fileReader.readLine();
    		
    		// Now read the file line by line starting from the first line
    		while ((line = fileReader.readLine()) != null) {
    			// Get all tokens available in line
				String[] tokens = line.split("!!!!");
				if (tokens.length > 0) {
					Item item = new Item(tokens[0], tokens[1]);  // 0 for the image FileName index and 1 for the image Label index
					itemList.add(item);
				}
    		}
    	} catch (Exception e) {
    		log.info("!!! Data file not found !!!");
    		e.printStackTrace();
    	} finally {
    		try {
    			fileReader.close();
    		} catch (IOException ioe) {
    			log.info("!!! Error while closing fileReader !!!");
				ioe.printStackTrace();
    		}
    	}
    	
    	return itemList;
    }
    
    public String cleanUp1(String in) {
    	String tmpString = new String();
    	
    	for (int i = 0; i < in.length(); i++) {
    		if (in.charAt(i) != '_') {
    			tmpString = tmpString + in.charAt(i);
    		}
    	}
    	
    	return tmpString.replaceAll("EEnndd", "");
    }
	
	
    public String cleanUp2(String in) {
    	
    	String cleaned =  new String();
    	String tmpString = new String();
    	
    	for (int i = 0; i < in.length(); i++) {
    		if (in.charAt(i) != '_') {
    			tmpString = tmpString + in.charAt(i);
    		}
    	}
    	
    	for (int i = (tmpString.length() - 1); i >= 0; i--) {
    		cleaned = cleaned + Character.toString(tmpString.charAt(i));
    	}
    	
    	return cleaned;
    }
}
