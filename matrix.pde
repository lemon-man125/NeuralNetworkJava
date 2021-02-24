float sigmoid(int v) {
    return 1 / (1 + exp(-v));
}

float sigmoid(float v) {
    return 1 / (1 + exp(-v));
}

float dsigmoid(float y) {
  return y * (1 - y);
}

float nMutate(float v, float rate) {
      if (random(1) < rate) {
        // return 2 * Math.random() - 1;
        return v + randomGaussian();
      } else {
        return v;
      }
    }

public class Matrix {
  public int rows;
  public int cols;
  public float[][] data;

  Matrix(int rows_, int cols_) {
    rows = rows_;
    cols = cols_;
    data = new float[rows][cols];
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        data[i][j] = 0;
      }
    }
  }
  
  Matrix(float[] arr) {
    rows = arr.length;
    cols = 1;
    data = new float[rows][cols];
    for (int i = 0; i < arr.length; i++) {
      data[i][0] = arr[i];
    }
  }
  
  Matrix(ArrayList<Float> arr) {
    rows = arr.size();
    cols = 1;
    data = new float[rows][cols];
    for (int i = 0; i < arr.size(); i++) {
      data[i][0] = arr.get(i);
    }
  } 

  // Adding functions for Neural Network
  
  public void mutate(float rate) {
   for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        data[i][j] = nMutate(data[i][j], rate);
      }
    }
  }
  
  public void dataSigmoid() {
   for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        data[i][j] = sigmoid(data[i][j]);
      }
    }
  }
  
  public void dataDSigmoid() {
   for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        data[i][j] = dsigmoid(data[i][j]);
      }
    }
  }
  
  // Normal Matrix Functions
  
  public Matrix copy() {
    Matrix result = new Matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        result.data[i][j] = data[i][j];
      }
    }
    return result;
  }
  
  public void randomize() {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        data[i][j] = random(-1, 1);
      }
    }
  }
  
  public Matrix transpose() {
    Matrix result = new Matrix(cols, rows);
     for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        result.data[j][i] = data[i][j];
      }
    }
    return result;
  }

  public void add(float n) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        data[i][j] += n;
      }
    }
  }

  public void add(Matrix m) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        data[i][j] += m.data[i][j];
      }
    }
  }
  
    public void sub(float n) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        data[i][j] -= n;
      }
    }
  }

  public void sub(Matrix m) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        data[i][j] -= m.data[i][j];
      }
    }
  }
  
  public Matrix getSubbed(Matrix m) {
    Matrix result = new Matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        result.data[i][j] = data[i][j] - m.data[i][j];
      }
    }
    return result;
  }
  
  public ArrayList<Float> toArray() {
    ArrayList<Float> arr = new ArrayList<Float>();
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        arr.add(data[i][j]);
      }
    }
    return arr;
  }

  public void multiply(float n) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        data[i][j] *= n;
      }
    }
  }
  
  public Matrix dot(Matrix m) {
    if (cols != m.rows) {
      println("Incompatible Matrix Sizes!");
      return null;
    }
    
    Matrix result = new Matrix(rows, m.cols);
    for (int i = 0; i < result.rows; i++) {
      float sum = 0;
      for (int j = 0; j < result.cols; j++) {
        for (int k = 0; k < cols; k++) {
          sum += data[i][k] * m.data[k][j];
        }
        result.data[i][j] = sum;
      }
       
    }
   return result;
  }
  
  public void multiply(Matrix b) {
      for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
          data[i][j] *= b.data[i][j];
        }
      }
  }
  
  public Matrix getDSigmoid() {
    Matrix result = new Matrix(rows, cols);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
         result.data[i][j] = dsigmoid(data[i][j]);
      }
    }     
    return result;
  }
  
  public void print() {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
         println(data[i][j]);
      }
    }    
  }
}
