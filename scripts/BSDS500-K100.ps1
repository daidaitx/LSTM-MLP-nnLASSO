## [generating test instances and save them]

# import BSDS500 dataset
Write-Host "==============================================导入案例==============================================" -ForegroundColor Yellow
python .\optimizees\nnlasso_bsds500.py

# solve LASSO with FISTA and save the optimal objective value for each instance (5000 iterations are sufficient to obtain optimal objective)
Write-Host "====================================以5000步proj-FISTA作为基准值====================================" -ForegroundColor Yellow
python main.py --config ./configs/nnlasso.yaml --optimizer proj-FISTA --save-dir BSDS500-K100/nnLASSO-proj-FISTA-baseline --test --test-batch-size 50 --load-mat --save-sol --optimizee-dir ./optimizees/matdata/nnlasso-real --test-length 5000

## [training commands] 
Write-Host "================================================训练================================================" -ForegroundColor Yellow
python main.py --config ./configs/nnlasso.yaml --optimizer LSTM-MLP                                     --save-dir BSDS500-K100/nnLASSO-noprocessing
python main.py --config ./configs/nnlasso.yaml --optimizer LSTM-MLP-x-proj                              --save-dir BSDS500-K100/nnLASSO-x-proj
python main.py --config ./configs/nnlasso.yaml --optimizer LSTM-MLP-y-proj                              --save-dir BSDS500-K100/nnLASSO-y-proj
python main.py --config ./configs/nnlasso.yaml --optimizee nnLASSO-linear-penalty      --penalty 0.024  --save-dir BSDS500-K100/nnLASSO-linear-penalty
python main.py --config ./configs/nnlasso.yaml --optimizee nnLASSO-quadratic-penalty   --penalty 0.220  --save-dir BSDS500-K100/nnLASSO-quadratic-penalty
python main.py --config ./configs/nnlasso.yaml --optimizee nnLASSO-exponential-penalty --penalty 0.020  --save-dir BSDS500-K100/nnLASSO-exponential-penalty
python main.py --config ./configs/nnlasso.yaml --optimizer LSTM-MLP-ADMM --optimizee nnLASSO-ADMM --penalty 0 --input-dim 64 --output-dim 32 --save-dir BSDS500-K100/nnLASSO-ADMM

## [test commands]
Write-Host "================================================测试================================================" -ForegroundColor Yellow
python main.py --config ./configs/nnlasso.yaml --optimizer LSTM-MLP                                     --save-dir BSDS500-K100/nnLASSO-noprocessing        --test --test-batch-size 50 --load-mat --load-sol --optimizee-dir ./optimizees/matdata/nnlasso-real
python main.py --config ./configs/nnlasso.yaml --optimizer proj-FISTA                                   --save-dir BSDS500-K100/nnLASSO-proj-FISTA          --test --test-batch-size 50 --load-mat --load-sol --optimizee-dir ./optimizees/matdata/nnlasso-real
python main.py --config ./configs/nnlasso.yaml --optimizer LSTM-MLP-x-proj                              --save-dir BSDS500-K100/nnLASSO-x-proj              --test --test-batch-size 50 --load-mat --load-sol --optimizee-dir ./optimizees/matdata/nnlasso-real
python main.py --config ./configs/nnlasso.yaml --optimizer LSTM-MLP-y-proj                              --save-dir BSDS500-K100/nnLASSO-y-proj              --test --test-batch-size 50 --load-mat --load-sol --optimizee-dir ./optimizees/matdata/nnlasso-real
python main.py --config ./configs/nnlasso.yaml --optimizee nnLASSO-linear-penalty      --penalty 0.024  --save-dir BSDS500-K100/nnLASSO-linear-penalty      --test --test-batch-size 50 --load-mat --load-sol --optimizee-dir ./optimizees/matdata/nnlasso-real
python main.py --config ./configs/nnlasso.yaml --optimizee nnLASSO-quadratic-penalty   --penalty 0.220  --save-dir BSDS500-K100/nnLASSO-quadratic-penalty   --test --test-batch-size 50 --load-mat --load-sol --optimizee-dir ./optimizees/matdata/nnlasso-real
python main.py --config ./configs/nnlasso.yaml --optimizee nnLASSO-exponential-penalty --penalty 0.020  --save-dir BSDS500-K100/nnLASSO-exponential-penalty --test --test-batch-size 50 --load-mat --load-sol --optimizee-dir ./optimizees/matdata/nnlasso-real
python main.py --config ./configs/nnlasso.yaml --optimizer LSTM-MLP-ADMM --optimizee nnLASSO-ADMM --penalty 0 --save-dir BSDS500-K100/nnLASSO-ADMM                --test --test-batch-size 50 --load-mat --load-sol --optimizee-dir ./optimizees/matdata/nnlasso-real

## [draw figures]
Write-Host "================================================绘图================================================" -ForegroundColor Yellow
python .\results\BSDS500-K100\draw_figure.py