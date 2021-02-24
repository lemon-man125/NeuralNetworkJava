public class NeuralNetwork {
  public int input_nodes;
  public int hidden_nodes;
  public int output_nodes;
  public float learning_rate;
  public int epochs;
  
  private Matrix weights_ih;
  private Matrix weights_ho;
  
  private Matrix bias_h;
  private Matrix bias_o;
  
  private ArrayList<float[][]> training_data = new ArrayList<float[][]>();
  //private ArrayList<float[][]> availableData = new ArrayList<float[][]>();

  
  NeuralNetwork(int inputs, int hidden, int outputs) {
    input_nodes = inputs;
    hidden_nodes = hidden;
    output_nodes = outputs;
    
    weights_ih = new Matrix(hidden_nodes, input_nodes);
    weights_ho = new Matrix(output_nodes, hidden_nodes);
    weights_ih.randomize();
    weights_ho.randomize();
    
    bias_h = new Matrix(hidden_nodes, 1);
    bias_o = new Matrix(output_nodes, 1);
    bias_h.randomize();
    bias_o.randomize();
    learning_rate = 0.1;
    epochs = 2;
    //for (float[][] data : training_data) {
    //  availableData.add(data);
    //}
    
  }
  
  NeuralNetwork(int inputs, int hidden, int outputs, float learning_rate_, int epochAmount) {
    input_nodes = inputs;
    hidden_nodes = hidden;
    output_nodes = outputs;
    
    weights_ih = new Matrix(hidden_nodes, input_nodes);
    weights_ho = new Matrix(output_nodes, hidden_nodes);
    weights_ih.randomize();
    weights_ho.randomize();
    
    bias_h = new Matrix(hidden_nodes, 1);
    bias_o = new Matrix(output_nodes, 1);
    bias_h.randomize();
    bias_o.randomize();
    
    epochs = epochAmount;
    
    learning_rate = learning_rate_;
    //for (float[][] data : training_data) {
    //  availableData.add(data);
    //}
  }
  
  NeuralNetwork(NeuralNetwork n) {
      input_nodes = n.input_nodes;
      hidden_nodes = n.hidden_nodes;
      output_nodes = n.output_nodes;

      weights_ih = n.weights_ih.copy();
      weights_ho = n.weights_ho.copy();

      bias_h = n.bias_h.copy();
      bias_o = n.bias_o.copy();
      
      setLearningRate(n.learning_rate);
      setEpochs(n.epochs);
  }
  
  public void setLearningRate(float rate) {
    learning_rate = rate;
  }
  
  public void setEpochs(int epochAmount) {
    epochs = epochAmount;
  }
  
  public void addData(float[] inputs_array, float[] target_array) {
    float[][] trainingArray = new float[2][];
    trainingArray[0] = inputs_array;
    trainingArray[1] = target_array;
    training_data.add(trainingArray);
    //availableData.add(trainingArray);
  }
  
  public ArrayList<Float> query(float[] inputs_array) {
    //if (inputs_array.length < input_nodes) {
    //  println("The length of the input's array that you passed in is NOT the same as the amount of input nodes");
    //  return null;
    //}
    Matrix inputs = new Matrix(inputs_array);
    
    // generate hidden outputs
    Matrix hidden = weights_ih.dot(inputs);
    hidden.add(bias_h);
    hidden.dataSigmoid();
    
    Matrix output = weights_ho.dot(hidden);
    output.add(bias_o);
    output.dataSigmoid();
    
    return output.toArray();
  }
  
   public void train() {
     for (int i = 0; i < epochs; i++) {
      for (int j = 0; j < training_data.size(); j++) {
      int index = floor(random(training_data.size()));
      float[][] data = training_data.get(index);
      float[] input_array = data[0];
      float[] target_array = data[1];
      // Generating the Hidden Outputs
      Matrix inputs = new Matrix(input_array);
      Matrix hidden = weights_ih.dot(inputs);
      hidden.add(bias_h);
      // activation function!
      hidden.dataSigmoid();
  
      // Generating the output's output!
   //   println(weights_ho.cols, hidden.rows);
      Matrix outputs = weights_ho.dot(hidden);
      outputs.add(bias_o);
      outputs.dataSigmoid();
  
      // Convert array to matrix object
      Matrix targets = new Matrix(target_array);
  
      // Calculate the error
      // ERROR = TARGETS - OUTPUTS
      Matrix output_errors = targets.getSubbed(outputs);
  
      // let gradient = outputs * (1 - outputs);
      // Calculate gradient
      Matrix gradients = outputs.getDSigmoid();
      gradients.multiply(output_errors);
      gradients.multiply(learning_rate);
  
  
      // Calculate deltas
      Matrix hidden_T = hidden.transpose();
      Matrix weight_ho_deltas = gradients.dot(hidden_T);
  
      // Adjust the weights by deltas
      weights_ho.add(weight_ho_deltas);
      // Adjust the bias by its deltas (which is just the gradients)
      bias_o.add(gradients);
  
      // Calculate the hidden layer errors
      Matrix who_t = weights_ho.transpose();
      Matrix hidden_errors = who_t.dot(output_errors);
  
      // Calculate hidden gradient
      Matrix hidden_gradient = hidden.getDSigmoid();
      hidden_gradient.multiply(hidden_errors);
      hidden_gradient.multiply(learning_rate);
  
      // Calcuate input->hidden deltas
      Matrix inputs_T = inputs.transpose();
      Matrix weight_ih_deltas = hidden_gradient.dot(inputs_T);
  
      weights_ih.add(weight_ih_deltas);
      // Adjust the bias by its deltas (which is just the gradients)
      bias_h.add(hidden_gradient);
  
      // outputs.print();
      // targets.print();
      // error.print();
      }
     }
  }
  
  // Adding functionality for neuro-evoulution
  public NeuralNetwork copy() {
    return new NeuralNetwork(this);
  }
  
  public void mutate(float rate) {
    weights_ih.mutate(rate);
    weights_ho.mutate(rate);
    bias_h.mutate(rate);
    bias_o.mutate(rate);
  }
}
