from engine.neuron import forward, forward_pass,backward_pass

lrs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
for lr in lrs:
    data = [(1, 1), (2, 4), (3, 9), (4, 16)]
    hidden_layer_weight1 = [[0.5, 0.3], [-0.2, 0.8], [0.1, -0.5]]
    hidden_layer_bias1      = [0.1, 0.2, 0.3]
    output_layer_weight = [[0.4, -0.1, 0.7]]
    output_bias  = [0.0]
    print(f"lr = {lr}")
    for i in range(1000):
        for x, actual in data:
            output1 = forward_pass([x], hidden_layer_weight1, hidden_layer_bias1)
            final_output = forward(output1, output_layer_weight[0], output_bias[0],False)
            
            hidden_layer1, hidden_layer_bias1, output_layer, output_layer  = backward_pass([x], output1, final_output,actual,hidden_layer_weight1,hidden_layer_bias1,output_layer_weight,output_bias, lr)
        # if i % 100 == 0:
        #     for x, actual in data:
        #         out1 = forward_pass([x, x*x], all_weights, biases)
        #         pred = forward(out1, weights2[0], biases2[0],False)
        #         print(f"x={x, x*x} | pred={pred:.2f} | actual={actual}")
        #     print("---")