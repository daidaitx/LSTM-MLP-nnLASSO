## [generating test instances and save them]

# generate LASSO instances and save them to "./optimizees/matdata/nnlasso-rand"
Write-Host "==============================================生成案例==============================================" -ForegroundColor Yellow
python main.py --config ./configs/nnlasso.yaml --optimizer proj-FISTA --save-dir synthetic-K100/nnLASSO-proj-FISTA-baseline --test --save-to-mat         --optimizee-dir ./optimizees/matdata/nnlasso-rand

# solve LASSO with FISTA and save the optimal objective value for each instance (5000 iterations are sufficient to obtain optimal objective)
Write-Host "====================================以5000步proj-FISTA作为基准值====================================" -ForegroundColor Yellow
python main.py --config ./configs/nnlasso.yaml --optimizer proj-FISTA --save-dir synthetic-K100/nnLASSO-proj-FISTA-baseline --test --load-mat --save-sol --optimizee-dir ./optimizees/matdata/nnlasso-rand --test-length 5000

## [training commands] 
Write-Host "================================================训练================================================" -ForegroundColor Yellow
python main.py --config ./configs/nnlasso.yaml --optimizer LSTM-MLP                                     --save-dir synthetic-K100/nnLASSO-noprocessing        --unroll-length 100
python main.py --config ./configs/nnlasso.yaml --optimizer LSTM-MLP-x-proj                              --save-dir synthetic-K100/nnLASSO-x-proj              --unroll-length 100
python main.py --config ./configs/nnlasso.yaml --optimizer LSTM-MLP-y-proj                              --save-dir synthetic-K100/nnLASSO-y-proj              --unroll-length 100
python main.py --config ./configs/nnlasso.yaml --optimizee nnLASSO-linear-penalty      --penalty 0.024  --save-dir synthetic-K100/nnLASSO-linear-penalty      --unroll-length 100
python main.py --config ./configs/nnlasso.yaml --optimizee nnLASSO-quadratic-penalty   --penalty 0.220  --save-dir synthetic-K100/nnLASSO-quadratic-penalty   --unroll-length 100
python main.py --config ./configs/nnlasso.yaml --optimizee nnLASSO-exponential-penalty --penalty 0.020  --save-dir synthetic-K100/nnLASSO-exponential-penalty --unroll-length 100
python main.py --config ./configs/nnlasso.yaml --optimizer LSTM-MLP-ADMM --input-dim 64 --output-dim 32 --save-dir synthetic-K100/nnLASSO-ADMM                --unroll-length 100

## [test commands]
Write-Host "================================================测试================================================" -ForegroundColor Yellow
python main.py --config ./configs/nnlasso.yaml --optimizer LSTM-MLP                                     --save-dir synthetic-K100/nnLASSO-noprocessing        --test-length 300 --test --load-mat --load-sol --optimizee-dir ./optimizees/matdata/nnlasso-rand
python main.py --config ./configs/nnlasso.yaml --optimizer proj-FISTA                                   --save-dir synthetic-K100/nnLASSO-proj-FISTA          --test-length 300 --test --load-mat --load-sol --optimizee-dir ./optimizees/matdata/nnlasso-rand
python main.py --config ./configs/nnlasso.yaml --optimizer LSTM-MLP-x-proj                              --save-dir synthetic-K100/nnLASSO-x-proj              --test-length 300 --test --load-mat --load-sol --optimizee-dir ./optimizees/matdata/nnlasso-rand
python main.py --config ./configs/nnlasso.yaml --optimizer LSTM-MLP-y-proj                              --save-dir synthetic-K100/nnLASSO-y-proj              --test-length 300 --test --load-mat --load-sol --optimizee-dir ./optimizees/matdata/nnlasso-rand
python main.py --config ./configs/nnlasso.yaml --optimizee nnLASSO-linear-penalty      --penalty 0.024  --save-dir synthetic-K100/nnLASSO-linear-penalty      --test-length 300 --test --load-mat --load-sol --optimizee-dir ./optimizees/matdata/nnlasso-rand
python main.py --config ./configs/nnlasso.yaml --optimizee nnLASSO-quadratic-penalty   --penalty 0.220  --save-dir synthetic-K100/nnLASSO-quadratic-penalty   --test-length 300 --test --load-mat --load-sol --optimizee-dir ./optimizees/matdata/nnlasso-rand
python main.py --config ./configs/nnlasso.yaml --optimizee nnLASSO-exponential-penalty --penalty 0.020  --save-dir synthetic-K100/nnLASSO-exponential-penalty --test-length 300 --test --load-mat --load-sol --optimizee-dir ./optimizees/matdata/nnlasso-rand
python main.py --config ./configs/nnlasso.yaml --optimizer LSTM-MLP-ADMM                                --save-dir synthetic-K100/nnLASSO-ADMM                --test-length 300 --test --load-mat --load-sol --optimizee-dir ./optimizees/matdata/nnlasso-rand

## [draw figures]
Write-Host "================================================绘图================================================" -ForegroundColor Yellow
python .\results\synthetic-K100\draw_figure.py