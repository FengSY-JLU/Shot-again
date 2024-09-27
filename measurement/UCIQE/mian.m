Files = dir(strcat('../clg60ablationSGE/J/clg60ablationSGE4280/','*.png'));
number=length(Files);
score = zeros(1, number);

for i=1:number
    disp(i);
    filename = Files(i).name;
    img = imread(['../clg60ablationSGE/J/clg60ablationSGE4280/',filename]);
    uciqe = UCIQE(img);
    uciqe = roundn(uciqe,-4);
    disp(filename);
    disp(uciqe);
    score(i) = uciqe;
end

mean_uicqe = mean(score);