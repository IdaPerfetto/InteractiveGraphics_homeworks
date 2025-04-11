// bgImg is the background image to be modified.
// fgImg is the foreground image.
// fgOpac is the opacity of the foreground image. The alpha value of the foreground image should be scaled using this argument. 
// fgPos is the position of the foreground image in pixels. It can be negative and (0,0) means the top-left pixels of the foreground and background are aligned.

function composite( bgImg, fgImg, fgOpac, fgPos )
{
    const bgData = bgImg.data;  // Pixel data in an array for the background image: each pixel has 4 values (RGBA) in the array
    const fgData = fgImg.data;  // Pixel data in an array for the foreground image
    const bgWidth = bgImg.width;  // Background image width
    const bgHeight = bgImg.height;  // Background image height
    const fgWidth = fgImg.width;  // Foreground image width
    const fgHeight = fgImg.height;  // Foreground image height
  
  // Let's iterate over each pixel of the foreground image
    for(let fgY = 0; fgY < fgHeight; fgY++){
      for(let fgX = 0; fgX < fgWidth; fgX++){
        // and calculate the corresponding pixel on the background by adding the position of the foreground image in pixels to the foreground's pixel coordinates
        let bgX = fgX + fgPos.x;
	let bgY = fgY + fgPos.y;
        
        // The parts of the foreground image that fall outside of the background image should be ignored:
        // if the coordinates we find are within the bounds of the background image
	if(bgX >= 0 && bgX < bgWidth && bgY >= 0 && bgY < bgHeight){
		// we define the index for the current background and foreground pixel
		let bgIdx = (bgY * bgWidth + bgX)*4; // multiplying by 4 to account for the 4 channels per pixel (RGBA)
	    	let fgIdx = (fgY * fgWidth + fgX)*4; 
	    	// and use it to extract the individual color channels and alpha values for both the foreground and background pixel:
		
		// cFgRed, cFgGreen, cFgBlue are the Red, Green, and Blue channels for the foreground pixel
		// cBgRed, cBgGreen, cBgBlue are the Red, Green, and Blue channels for the background pixel
		// alphaFg is the transparency for the foreground pixel (that has to be adjusted by the foreground opacity)
		// alphaBg is the transparency for the background pixel
          
		let cFgRed =  fgData[fgIdx];
		let cFgGreen =  fgData[fgIdx + 1];
		let cFgBlue =  fgData[fgIdx + 2];
		let cBgRed =  bgData[bgIdx];
		let cBgGreen =  bgData[bgIdx + 1];
		let cBgBlue =  bgData[bgIdx + 2];
		let alphaFg = (fgData[fgIdx + 3] * fgOpac)/255;	// normalized to the range [0,1]
		let alphaBg = (bgData[bgIdx + 3])/255; // normalized to the range [0,1]
		let alpha = alphaFg + (1- alphaFg)*alphaBg; // composite alpha value that results from blending the foreground and background pixel
		
		// Let's now process the pixel only if it's not fully transparent
		if(alpha > 0){
			
			// we have to blend the color values of the foreground image with the background image, based on their respective alpha values
			// the blendend color c is given by:
			// the contibution of the foreground color expressed as the foreground color multiplied by its alpha (alphaFg*cFg)
			// the contibution of the background color expressed as the background color multiplied by its alpha (alphaBg*cBg)
			// remembering that as the foreground becomes more opaque, less of the background should be visible (so we multiply (alphaBg*cBg) by (1-alphaFg))
			// and dividing by alpha to normalize
			bgData[bgIdx] = ((alphaFg*cFgRed) + (1-alphaFg)*alphaBg*cBgRed) / alpha;
			bgData[bgIdx + 1] = ((alphaFg*cFgGreen) + (1-alphaFg)*alphaBg*cBgGreen) / alpha;
			bgData[bgIdx + 2] = ((alphaFg*cFgBlue) + (1-alphaFg)*alphaBg*cBgBlue) / alpha;
			
			// we update the alpha channel for the background pixel is updated to the composite alpha value (alpha) multiplied by 255 to go back to the [0,255] range
			bgData[bgIdx + 3] = alpha * 255;
		}
	}
      }
    }
}
