function result_mvkdr(file_name)
repeat = 10;
th = [0.01, 0.1, 1, 10, 100];

obj = zeros(repeat, 5, 5);
acc = zeros(repeat, 5, 5);
nmi = zeros(repeat, 5, 5);
for i=1:repeat
    for j=1:5
        for k=1:5
            load(strjoin({file_name, '_seed', num2str(i-1), '_th1', num2str(th(j)), '_th2', num2str(th(k)), '.mat'}, ''));
            obj(i, j, k) = km_obj;
            [acc(i, j, k), nmi(i, j, k), ] = CalcMetrics(y', y_pred');
        end
    end
end
obj
obj = reshape(mean(obj), [5, 5]);
acc = reshape(mean(acc), [5, 5]);
nmi = reshape(mean(nmi), [5, 5]);
[M, I] = min(obj(:));
[I_row, I_col] = ind2sub(size(obj),I);
acc = acc(I_row, I_col);
nmi = nmi(I_row, I_col);
save(file_name, 'acc', 'nmi');
exit();
