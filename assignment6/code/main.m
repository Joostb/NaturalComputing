addpath src/adrem
addpath src/evaluation
data = load_dataset('office-caltech', 'surf');
% Predict labels of 'dvd' target domain using 'books' as source
% Preprocess data: divide by standard deviation over both domains
results = zeros(5,4);
percentage = [10, 20, 40, 60];

for p = 1:4
    for i = 1:5 
        idx = randperm(length(data.x{2}));
        indexToGroup1 = (idx <= length(data.x{2})/100*percentage(p));
        indexToGroup2 = (idx > length(data.x{2})/100*percentage(p));
        group1 = data.x{2}(indexToGroup1, :);
        group2 = data.x{2}(indexToGroup2, :);
        label1 = data.y{2}(indexToGroup1, :);
        label2 = data.y{2}(indexToGroup2, :);

        source_x = [data.x{1}; group1];
        source_y = [data.y{1}; label1];
        target_x = group2;
        target_y = label2;
        [x_src, x_tgt] = preprocess(source_x, source_y, target_x, 'joint-std');
        y = predict_adrem(x_src, source_y, x_tgt);
        results(i, p) = mean(y == target_y);
    end 
end

mean(results)

%% Plotting
boxplot(results)
xlabel('Percentage')
ylabel('Accuracy')
xticklabels({'10', '20', '40', '60'})
saveas(gcf, 'adrem_results.png')