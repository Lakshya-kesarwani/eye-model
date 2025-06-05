import * as fs from 'fs';
import * as path from 'path';
import { getCluster } from "./gridutils"; // Add './' to specify the relative path

// Path to the dataset folder
const DATASET_DIR = 'dataset_cropped/images';
const OUTPUT_DIR = 'clustered_images';

/**
 * Parses the filename to extract the label.
 */
function parseFilename(filename: string): number | null {
  const match = filename.match(/^(\d+)_/); // Match filenames like "0_xcoord_ycoord_time.jpg"
  if (!match) return null;

  return parseInt(match[1], 10); // Extract the label as a number
}

/**
 * Processes the dataset folder, classifies images into clusters, and saves them.
 */
export function processAndSaveImages() {
  const files = fs.readdirSync(DATASET_DIR);

  files.forEach((file) => {
    const label = parseFilename(file);
    if (label !== null) {
      const cluster = getCluster(label);

      // Create the cluster folder if it doesn't exist
      const clusterFolder = path.join(OUTPUT_DIR, cluster);
      if (!fs.existsSync(clusterFolder)) {
        fs.mkdirSync(clusterFolder, { recursive: true });
      }

      // Copy the image to the cluster folder
      const sourcePath = path.join(DATASET_DIR, file);
      const destinationPath = path.join(clusterFolder, file);
      fs.copyFileSync(sourcePath, destinationPath);

      console.log(`Image ${file} classified to cluster ${cluster}`);
    } else {
      console.warn(`Skipping invalid file: ${file}`);
    }
  });
}

// Run the function to process and save images
processAndSaveImages();