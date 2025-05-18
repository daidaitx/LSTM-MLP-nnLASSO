## [generating test instances and save them]

# generate LASSO instances and save them to "./optimizees/matdata/nnlasso-rand"
Write-Host "==============================================生成案例==============================================" -ForegroundColor Yellow
python main.py --config ./configs/nnlasso.yaml --optimizer proj-FISTA --save-dir synthetic-K20-penalty/nnLASSO-proj-FISTA-baseline --test --save-to-mat         --optimizee-dir ./optimizees/matdata/nnlasso-rand

# solve LASSO with FISTA and save the optimal objective value for each instance (5000 iterations are sufficient to obtain optimal objective)
Write-Host "====================================以5000步proj-FISTA作为基准值====================================" -ForegroundColor Yellow
python main.py --config ./configs/nnlasso.yaml --optimizer proj-FISTA --save-dir synthetic-K20-penalty/nnLASSO-proj-FISTA-baseline --test --load-mat --save-sol --optimizee-dir ./optimizees/matdata/nnlasso-rand --test-length 5000

## [training commands] 
Write-Host "================================================训练================================================" -ForegroundColor Yellow
python main.py --config ./configs/nnlasso.yaml --optimizee nnLASSO-linear-penalty      --penalty 0.010  --save-dir synthetic-K20-penalty/nnLASSO-linear-penalty-0.01
python main.py --config ./configs/nnlasso.yaml --optimizee nnLASSO-linear-penalty      --penalty 0.020  --save-dir synthetic-K20-penalty/nnLASSO-linear-penalty-0.02
python main.py --config ./configs/nnlasso.yaml --optimizee nnLASSO-linear-penalty      --penalty 0.030  --save-dir synthetic-K20-penalty/nnLASSO-linear-penalty-0.03
python main.py --config ./configs/nnlasso.yaml --optimizee nnLASSO-quadratic-penalty   --penalty 0.100  --save-dir synthetic-K20-penalty/nnLASSO-quadratic-penalty-0.1
python main.py --config ./configs/nnlasso.yaml --optimizee nnLASSO-quadratic-penalty   --penalty 0.200  --save-dir synthetic-K20-penalty/nnLASSO-quadratic-penalty-0.2
python main.py --config ./configs/nnlasso.yaml --optimizee nnLASSO-quadratic-penalty   --penalty 0.300  --save-dir synthetic-K20-penalty/nnLASSO-quadratic-penalty-0.3
python main.py --config ./configs/nnlasso.yaml --optimizee nnLASSO-exponential-penalty --penalty 0.010  --save-dir synthetic-K20-penalty/nnLASSO-exponential-penalty-0.01
python main.py --config ./configs/nnlasso.yaml --optimizee nnLASSO-exponential-penalty --penalty 0.020  --save-dir synthetic-K20-penalty/nnLASSO-exponential-penalty-0.02
python main.py --config ./configs/nnlasso.yaml --optimizee nnLASSO-exponential-penalty --penalty 0.030  --save-dir synthetic-K20-penalty/nnLASSO-exponential-penalty-0.03

## [test commands]
Write-Host "================================================测试================================================" -ForegroundColor Yellow
python main.py --config ./configs/nnlasso.yaml --optimizee nnLASSO-linear-penalty      --penalty 0.010  --save-dir synthetic-K20-penalty/nnLASSO-linear-penalty-0.01        --test --load-mat --load-sol --optimizee-dir ./optimizees/matdata/nnlasso-rand
python main.py --config ./configs/nnlasso.yaml --optimizee nnLASSO-linear-penalty      --penalty 0.020  --save-dir synthetic-K20-penalty/nnLASSO-linear-penalty-0.02        --test --load-mat --load-sol --optimizee-dir ./optimizees/matdata/nnlasso-rand
python main.py --config ./configs/nnlasso.yaml --optimizee nnLASSO-linear-penalty      --penalty 0.030  --save-dir synthetic-K20-penalty/nnLASSO-linear-penalty-0.03        --test --load-mat --load-sol --optimizee-dir ./optimizees/matdata/nnlasso-rand
python main.py --config ./configs/nnlasso.yaml --optimizee nnLASSO-quadratic-penalty   --penalty 0.100  --save-dir synthetic-K20-penalty/nnLASSO-quadratic-penalty-0.1      --test --load-mat --load-sol --optimizee-dir ./optimizees/matdata/nnlasso-rand
python main.py --config ./configs/nnlasso.yaml --optimizee nnLASSO-quadratic-penalty   --penalty 0.200  --save-dir synthetic-K20-penalty/nnLASSO-quadratic-penalty-0.2      --test --load-mat --load-sol --optimizee-dir ./optimizees/matdata/nnlasso-rand
python main.py --config ./configs/nnlasso.yaml --optimizee nnLASSO-quadratic-penalty   --penalty 0.300  --save-dir synthetic-K20-penalty/nnLASSO-quadratic-penalty-0.3      --test --load-mat --load-sol --optimizee-dir ./optimizees/matdata/nnlasso-rand
python main.py --config ./configs/nnlasso.yaml --optimizee nnLASSO-exponential-penalty --penalty 0.010  --save-dir synthetic-K20-penalty/nnLASSO-exponential-penalty-0.01   --test --load-mat --load-sol --optimizee-dir ./optimizees/matdata/nnlasso-rand
python main.py --config ./configs/nnlasso.yaml --optimizee nnLASSO-exponential-penalty --penalty 0.020  --save-dir synthetic-K20-penalty/nnLASSO-exponential-penalty-0.02   --test --load-mat --load-sol --optimizee-dir ./optimizees/matdata/nnlasso-rand
python main.py --config ./configs/nnlasso.yaml --optimizee nnLASSO-exponential-penalty --penalty 0.030  --save-dir synthetic-K20-penalty/nnLASSO-exponential-penalty-0.03   --test --load-mat --load-sol --optimizee-dir ./optimizees/matdata/nnlasso-rand

## [draw figures]
Write-Host "================================================绘图================================================" -ForegroundColor Yellow
python .\results\synthetic-K20-penalty\draw_figure.py