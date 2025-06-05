import * as FileSystem from 'expo-file-system';

/**
 * Saves an image with the given label and coordinates.
 */
export async function saveImage(label: string, x: number, y: number, imageData: string) {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  const filename = `${label}_${x}_${y}_${timestamp}.jpg`;
  const filePath = `${FileSystem.documentDirectory}${filename}`;

  try {
    await FileSystem.writeAsStringAsync(filePath, imageData, {
      encoding: FileSystem.EncodingType.Base64,
    });
    console.log(`Image saved: ${filePath}`);
  } catch (error) {
    console.error('Error saving image:', error);
  }
}