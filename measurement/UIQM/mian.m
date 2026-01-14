Files = dir(strcat('../clg60ablationSGE/J/clg60ablationSGE4280/','*.png'));
number=length(Files);
score = zeros(1, number);
score_uicm = zeros(1, number);

for i=1:number
    disp(i);
    filename = Files(i).name;
    img = imread(['../clg60ablationSGE/J/clg60ablationSGE4280/',filename]);
    uiqm = UIQM(img);
    uiqm = roundn(uiqm,-4);
    disp(filename);
    disp(uiqm);
    score(i) = uiqm;
    uicm = UICM(img);
    uicm = roundn(uicm,-4);
    score_uicm(i) = uicm; 
end

mean_uiqm = mean(score);



