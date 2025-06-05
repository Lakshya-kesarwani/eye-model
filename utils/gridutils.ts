export const GRID_CONFIG = {
    rows: 9, // 9x9 grid
    cols: 9,
  };
  
  export const CLUSTER_CONFIG = {
    rows: 3, // 3x3 clusters
    cols: 3,
    labels: ['A', 'B', 'C', 'D','BKL', 'E', 'F', 'G', 'H'], // 8 clusters
  };
  
  /**
   * Maps a label (0–81) to its corresponding cluster (A–H).
   */
  export function getCluster(label: number) {
    const row = Math.floor(label / GRID_CONFIG.cols); // Calculate the row index
    const col = label % GRID_CONFIG.cols; // Calculate the column index
  
    // Determine the cluster based on the 3x3 grid
    const clusterRow = Math.floor(row / CLUSTER_CONFIG.rows);
    const clusterCol = Math.floor(col / CLUSTER_CONFIG.cols);
  
    const clusterIndex = clusterRow * (GRID_CONFIG.cols / CLUSTER_CONFIG.cols) + clusterCol;
    return clusterIndex;
  }
