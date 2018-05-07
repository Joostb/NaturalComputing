addpath src/adrem
addpath src/evaluation
data = load_dataset('office-caltech');
% Predict labels of 'dvd' target domain using 'books' as source
% Preprocess data: divide by standard deviation over both domains
[x_src, x_tgt] = preprocess(data.x{1}, data.y{1}, data.x{2}, 'joint-std');
y = predict_adrem(x_src, data.y{1}, x_tgt);
mean(y == data.y{2})