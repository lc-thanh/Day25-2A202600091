# Báo cáo GPU FinOps Lab

**Họ tên:** Lê Công Thành  
**Mã HV:** 2A202600091  
**Ngày thực hiện:** 13/05/2026

---

## 1. Giới thiệu

### 1.1. Mục tiêu của bài lab

Bài lab GPU FinOps có mục tiêu mô phỏng và thực hành quy trình quản lý chi phí hạ tầng GPU cho các workload AI/ML. Thông qua hệ thống gồm nhiều service nhỏ chạy bằng Docker Compose và notebook thực thi trên Kaggle/Colab, bài lab giúp quan sát cách GPU được cấp phát, cách workload tạo ra chi phí, cách spot instance tạo ra savings, cách autoscaler phản ứng với mức sử dụng tài nguyên, và cách phân tích lãng phí để đưa ra khuyến nghị tối ưu.

Các mục tiêu chính gồm:

- Theo dõi trạng thái GPU cluster, số lượng GPU bận/rảnh, mức sử dụng bộ nhớ, công suất và nhiệt độ.
- Submit workload giả lập và ghi nhận chi phí theo loại GPU, thời lượng chạy, số GPU sử dụng và hình thức on-demand/spot.
- Phân tích lợi ích của spot instance, bao gồm chi phí thấp hơn và rủi ro preemption.
- Đánh giá chính sách autoscaling theo hướng KEDA-like dựa trên utilization threshold.
- Tạo cost snapshot, waste report, dashboard và recommendation để hỗ trợ quyết định FinOps.
- Thực thi workload thật trên GPU Kaggle, so sánh FP32 với Mixed Precision AMP về thời gian, bộ nhớ, utilization và chi phí.
- Thực hiện các phân tích nâng cao như multi-GPU scaling, project cost forecasting và thiết kế chiến lược tối ưu chi phí.

### 1.2. Tổng quan về GPU FinOps

GPU FinOps là cách tiếp cận kết hợp giữa kỹ thuật, tài chính và vận hành nhằm tối ưu chi phí sử dụng GPU mà vẫn đảm bảo hiệu năng và yêu cầu dự án. Với workload AI/ML, GPU thường là thành phần chi phí cao nhất vì giá thuê theo giờ của các GPU như A100, V100 hoặc T4 có thể tăng nhanh khi training kéo dài hoặc khi cluster bị cấp phát dư thừa.

Một quy trình GPU FinOps hiệu quả cần trả lời được các câu hỏi:

- GPU nào đang được sử dụng, GPU nào đang idle?
- Workload nào tiêu tốn nhiều chi phí nhất?
- Có thể dùng spot instance cho workload fault-tolerant hay không?
- Có thể giảm thời gian training bằng Mixed Precision, batch-size tuning hoặc early stopping không?
- Có nên scale up/down cluster tại thời điểm hiện tại không?
- Chi phí dự án trong tương lai nằm trong khoảng nào và cần contingency bao nhiêu?

Trong bài lab này, các câu hỏi trên được mô phỏng bằng một nền tảng gồm gateway, GPU node manager, billing API, spot manager, autoscaler và cost tracker. Notebook đóng vai trò client để gọi API, trực quan hóa kết quả và thực hiện training thực tế trên GPU.

---

## 2. Phân tích từng phần

## Part 1-7: Phân tích kết quả từ mock GPU cluster

### Part 1. GPU Cluster Monitoring

Ở trạng thái ban đầu, cluster có **4 nodes** với tổng cộng **8 GPU**. Các GPU gồm nhiều loại khác nhau: **T4**, **A100** và **V100**. Kết quả Cell 3 cho thấy toàn bộ GPU ban đầu đều ở trạng thái idle, nhưng vẫn có mức utilization nền nhỏ và vẫn tiêu thụ điện.

Thông tin tổng hợp từ Cell 4:

| Chỉ số | Giá trị |
|---|---:|
| Tổng số GPU | 8 |
| Busy GPU | 0 |
| Idle GPU | 8 |
| Average utilization | 8.6% |
| Memory used | 9.0 GB |
| Memory capacity | 288.0 GB |
| Total power draw | 265 W |
| Node count | 4 |

Nhận xét: ngay cả khi chưa có workload nào chạy, cluster vẫn tiêu thụ **265 W** và có utilization trung bình **8.6%**. Đây là ví dụ rõ ràng cho hiện tượng idle cost trong GPU FinOps: tài nguyên đã được provision nhưng chưa tạo ra giá trị trực tiếp. Nếu trạng thái này kéo dài trong môi trường thật, chi phí sẽ tiếp tục phát sinh dù GPU không phục vụ workload quan trọng.

### Part 2. Workload Submission & Cost Tracking

Sau khi submit workload, notebook đã chạy 4 workload:

| Workload | Trạng thái | GPU được cấp phát |
|---|---|---|
| `train-resnet-001` | running | 1 GPU trên node-00 |
| `train-bert-002` | running | 1 GPU trên node-01 |
| `inference-api-003` | running | 1 GPU trên node-00 |
| `train-llm-004` | running | 2 GPU trên node-01 và node-02 |

Sau bước này, cluster tăng lên **5 busy GPU / 8 total GPU**, utilization đạt **49.2%**. Điều này cho thấy workload submission đã giúp tài nguyên được sử dụng hiệu quả hơn so với trạng thái ban đầu, nhưng vẫn còn **3 GPU idle**.

Kết quả billing Cell 6:

| Workload | Hình thức | Chi phí | Savings |
|---|---|---:|---:|
| `train-resnet-001` | On-demand | $0.0292 | $0.0000 |
| `train-bert-002` | On-demand | $0.6117 | $0.0000 |
| `inference-api-003` | Spot | $0.0035 | $0.0082 |
| `train-llm-004` | Spot | $0.5505 | $1.2845 |

Tổng kết billing:

- **Total Cost:** $1.1949
- **Total Savings:** $1.2927
- **Budget Used:** 1.2%
- **Alert Status:** OK

Nhận xét: workload `train-llm-004` dùng 2 GPU và tạo ra chi phí lớn nhất trong nhóm workload, nhưng vì chạy ở chế độ spot nên tiết kiệm được **$1.2845**. Điều này thể hiện vai trò quan trọng của việc phân loại workload: các workload training có khả năng retry/checkpoint thường phù hợp để chuyển sang spot nhằm giảm chi phí.

### Part 3. Spot Instance Management

Cell 7 hiển thị bảng giá spot hiện tại:

| GPU Type | On-Demand | Spot Price | Discount | Availability |
|---|---:|---:|---:|---|
| T4 | $0.35 | $0.2433 | 30.5% | low |
| A100 | $3.67 | $2.8531 | 22.3% | medium |
| V100 | $2.48 | $1.8325 | 26.1% | high |

Cell 8 request 3 spot instances và cả 3 đều được cấp phát thành công:

- `spot-t4-001`: granted
- `spot-t4-002`: granted
- `spot-a100-001`: granted

Cell 9 mô phỏng preemption và kết quả lần chạy này không có instance nào bị preempted:

- **Preempted instances:** 0
- **Still active:** 3
- **Spot cost:** $0.0003
- **On-demand equivalent:** $0.0009
- **Total saved:** $0.0006, tương đương **61.8%**

Nhận xét: spot instance đem lại mức tiết kiệm cao, trong lần mô phỏng này đạt **61.8%**. Tuy nhiên, kết quả availability cho thấy T4 có availability thấp, A100 trung bình và V100 cao. Do đó, spot nên được dùng cho workload có khả năng checkpoint, retry hoặc không yêu cầu SLA nghiêm ngặt. Với workload inference production hoặc training không thể gián đoạn, cần cân nhắc giữa savings và rủi ro preemption.

### Part 4. Autoscaling KEDA-like

Chính sách autoscaling ban đầu có các tham số:

| Tham số | Giá trị ban đầu |
|---|---:|
| scale_up_threshold | 80.0 |
| scale_down_threshold | 20.0 |
| cooldown_seconds | 60 |
| max_nodes | 8 |
| min_nodes | 1 |
| preferred_gpu_type | T4 |
| cost_aware | True |

Sau khi update policy, autoscaler đánh giá cluster với utilization **49.2%**. Kết quả là **NO_ACTION**, vì utilization nằm trong khoảng threshold mới **25.0% - 70.0%**. Trong 5 evaluation cycles, autoscaler đều giữ nguyên số node:

| Cycle | Action | Utilization | Nodes |
|---|---|---:|---|
| 1 | no_action | 49.2% | 4 → 4 |
| 2 | no_action | 49.2% | 4 → 4 |
| 3 | no_action | 49.2% | 4 → 4 |
| 4 | no_action | 49.2% | 4 → 4 |
| 5 | no_action | 49.2% | 4 → 4 |

Nhận xét: autoscaler hoạt động đúng logic threshold. Khi utilization chưa vượt ngưỡng scale up và cũng chưa thấp đến mức scale down, hệ thống không thay đổi số node. Đây là hành vi mong muốn vì scale không cần thiết có thể làm tăng chi phí hoặc gây nhiễu vận hành. Việc bật `cost_aware=True` cũng phù hợp với mục tiêu FinOps: autoscaling không chỉ tối ưu hiệu năng mà còn phải cân nhắc chi phí.

### Part 5. Cost Analysis & Optimization

Cell 12 tạo 5 cost snapshots. Các snapshot đều có cùng giá trị:

| Snapshot | Total Cost | Idle Cost | Waste |
|---|---:|---:|---:|
| 1 | $0.038056 | $0.008833 | 23.2% |
| 2 | $0.038056 | $0.008833 | 23.2% |
| 3 | $0.038056 | $0.008833 | 23.2% |
| 4 | $0.038056 | $0.008833 | 23.2% |
| 5 | $0.038056 | $0.008833 | 23.2% |

Waste report Cell 13:

- **Average Waste:** 23.2%
- **Total Idle Cost:** $0.044165
- **Total Cost:** $0.190280
- **Potential Monthly Save:** $2,289.51
- **Severity:** LOW

Nhận xét: mức waste **23.2%** được phân loại LOW trong lab, nhưng nếu extrapolate theo tháng thì potential monthly saving lên tới **$2,289.51**. Điều này cho thấy một tỷ lệ waste tưởng như không quá nghiêm trọng vẫn có thể trở thành khoản chi đáng kể khi workload chạy liên tục hoặc khi quy mô cluster tăng.

Cell 14 đưa ra hai recommendation:

1. **USE_SPOT** - mức độ Medium, khuyến nghị chuyển workload fault-tolerant sang spot instance để tiết kiệm khoảng **65%**.
2. **SCHEDULING** - mức độ Low, khuyến nghị chạy job không khẩn cấp vào off-peak hours để tiết kiệm khoảng **20%**.

Các khuyến nghị này phù hợp với dữ liệu đã quan sát: spot đem lại savings cao, còn scheduling giúp giảm chi phí mà ít ảnh hưởng đến kiến trúc hệ thống.

### Part 6. Visualization

Cell 16 tạo file `finops_cost_breakdown.png`, gồm 3 biểu đồ mô tả breakdown chi phí. Cell 17 tạo time-series cost tracking với stackplot và waste percentage. Các file được lưu trong thư mục `generated_charts/`.

Các visualization giúp biến số liệu thô thành thông tin dễ hiểu hơn:

- Cost breakdown cho thấy mối quan hệ giữa on-demand cost, spot cost, savings và waste.
- Time-series tracking giúp quan sát xu hướng chi phí theo thời gian.
- Waste percentage giúp phát hiện thời điểm tài nguyên bị cấp phát dư thừa.

Nhận xét: trong thực tế, dashboard dạng này rất hữu ích cho nhóm vận hành và nhóm tài chính vì giúp theo dõi chi phí GPU liên tục, thay vì chỉ xem hóa đơn cuối kỳ.

### Part 7. Complete FinOps Workflow

Cell 18 mô phỏng toàn bộ workflow tối ưu GPU FinOps:

1. Trạng thái ban đầu: **8 GPU**, utilization **49.2%**, idle **3 GPU**.
2. Submit heavy workloads: utilization tăng lên **75.3%**, busy **8/8 GPU**.
3. Autoscaler evaluation: quyết định **scale_up** vì utilization **75.3% > threshold 70.0%**.
4. Cost analysis: total cost/interval **$0.040000**, waste **4.9%**.
5. Recommendations: USE_SPOT tiết kiệm khoảng **65%**, SCHEDULING tiết kiệm khoảng **20%**.
6. Applying optimization: chuyển sang spot instances, spot savings **$0.0367 (70.0%)**.
7. Final billing: total spend **$1.3640**, total saved **$1.4151**, budget used **1.4%**.

Nhận xét: workflow này thể hiện vòng lặp FinOps hoàn chỉnh: quan sát tài nguyên, tạo tải, đánh giá autoscaling, phân tích chi phí, đưa ra recommendation, áp dụng optimization và kiểm tra lại billing. Kết quả cho thấy khi cluster được sử dụng gần hết, waste giảm từ **23.2%** xuống **4.9%**, đồng thời việc dùng spot tiếp tục tạo savings đáng kể.

---

## Part 8: Phân tích real GPU training trên Kaggle

### 8.1. GPU environment và diagnostic

Notebook đã phát hiện GPU thật trên Kaggle:

| Thuộc tính | Giá trị |
|---|---|
| GPU name | Tesla T4 |
| Memory | 15.6 GB |
| GPU type | T4 |
| Pricing | $0.35/hr on-demand |
| CUDA | 12.8 |
| pynvml | available |

Diagnostic Cell 20 cho thấy `pynvml` hoạt động, đọc được thông tin GPU như memory, power và temperature. Trước training, GPU gần như idle:

- GPU utilization: 0%
- Memory used: 472 MB / 16106 MB
- Power: 9.9 W
- Temperature: 40°C

Điều này xác nhận môi trường Kaggle đã bật GPU đúng cách và notebook có thể thu thập telemetry trong lúc training.

### 8.2. FP32 baseline training

Cell 22 chạy training FP32 trên CIFAR-10 với ResNet-18 trong 3 epochs. Kết quả:

| Epoch | Loss | Accuracy | Time |
|---|---:|---:|---:|
| 1 | 2.1147 | 25.1% | 39.2s |
| 2 | 1.5145 | 44.3% | 38.9s |
| 3 | 1.2200 | 56.1% | 40.3s |

FP32 summary:

- **Total time:** 118.4s
- **Peak memory:** 0.82 GB
- **Average GPU utilization:** 95.1%
- **Average power:** 66.7 W
- **Average temperature:** 64.0°C
- **Max GPU utilization:** 98.0%
- **Estimated cost:** $0.011507

Nhận xét: FP32 sử dụng GPU rất cao, average utilization đạt **95.1%**, cho thấy workload đã tận dụng tốt tài nguyên tính toán. Tuy nhiên, thời gian chạy dài hơn AMP nên chi phí tổng thể cao hơn.

### 8.3. Mixed Precision AMP training

Cell 23 chạy cùng workload bằng Mixed Precision AMP. Kết quả:

| Epoch | Loss | Accuracy | Time |
|---|---:|---:|---:|
| 1 | 2.0883 | 27.0% | 19.3s |
| 2 | 1.4703 | 45.4% | 18.6s |
| 3 | 1.1824 | 57.4% | 18.5s |

AMP summary:

- **Total time:** 56.4s
- **Peak memory:** 0.60 GB
- **Average GPU utilization:** 90.8%
- **Average power:** 65.3 W
- **Average temperature:** 77.7°C
- **Max GPU utilization:** 93.0%
- **Estimated cost:** $0.005479

Nhận xét: AMP giảm thời gian training xuống còn **56.4s**, tức nhanh hơn đáng kể so với FP32. Peak memory cũng giảm từ **0.82 GB** xuống **0.60 GB**. Accuracy cuối đạt **57.4%**, tương đương và thậm chí cao hơn nhẹ so với FP32 trong lần chạy này, cho thấy AMP không làm giảm chất lượng kết quả trong bài thử nghiệm.

### 8.4. So sánh FP32 vs AMP theo góc nhìn FinOps

Kết quả Cell 24:

| Metric | FP32 | AMP | Cải thiện |
|---|---:|---:|---:|
| Total Time | 118.4s | 56.4s | 2.10x faster |
| Peak Memory | 0.82 GB | 0.60 GB | 0.22 GB saved |
| Cost | $0.011507 | $0.005479 | $0.006028 saved |
| Cost Saving | - | - | 52.4% |
| Avg GPU Utilization | 95.1% | 90.8% | - |
| Avg Power | 66.7 W | 65.3 W | - |

Extrapolated savings:

| Quy mô training | FP32 | AMP | Savings |
|---|---:|---:|---:|
| 1 ngày | $8.40 | $4.00 | $4.40 |
| 1 tuần | $58.80 | $28.00 | $30.80 |
| 1 tháng | $252.00 | $119.99 | $132.01 |

Nhận xét: Mixed Precision AMP là một optimization có tác động lớn vì vừa giảm thời gian training, vừa giảm memory footprint, vừa giảm chi phí. Mức tiết kiệm **52.4%** trên bài thử nhỏ cho thấy nếu áp dụng ở quy mô lớn, ví dụ training liên tục trong một tháng, savings có thể đạt hơn **$132** chỉ với một GPU T4. Với nhiều GPU hoặc GPU đắt hơn như A100, tác động tài chính sẽ còn lớn hơn.

### 8.5. Report real GPU costs back to gateway

Cell 25 gửi chi phí real GPU về FinOps gateway:

- FP32 workload reported: cost **$0.011500**, rate **$0.3500/hr**.
- AMP workload reported as spot: cost **$0.001600**, saved **$0.003800**.
- Project `real-gpu-lab`: total cost **$0.013100**, total savings **$0.003800**, workloads **2**.
- Cost snapshot after real GPU report: waste **22.1%**.
- Final platform dashboard: total platform cost **$1.3640**, total savings **$1.4151**, budget utilization **1.4%**, alert **OK**.

Nhận xét: việc gửi kết quả training thật về gateway giúp hợp nhất mock cluster cost và real GPU cost trong cùng dashboard. Đây là mô hình gần với thực tế, nơi các nguồn workload khác nhau cần được chuẩn hóa và gom về một hệ thống FinOps chung.

---

## Part 8.5: Advanced GPU Cost Optimization

### 8.5.1. Multi-GPU Cost Analysis

Cell 27 phân tích scaling cho GPU T4 với base time **2.0 hours** và giá **$0.35/hr**. Kết quả notebook xác định:

- **Most cost-efficient:** 8 GPU, với **$0.19 per performance unit**.
- **Fastest configuration:** 8 GPU, hoàn thành trong **0.36 hours**.

Nhận xét: tăng số GPU có thể giảm thời gian hoàn thành, nhưng hiệu quả không tuyến tính tuyệt đối do scaling efficiency giảm dần. Trong lần phân tích này, cấu hình 8 GPU vẫn là lựa chọn tốt nhất về cost efficiency và tốc độ. Điều này cho thấy quyết định số lượng GPU không nên chỉ dựa vào chi phí theo giờ, mà cần xét cả thời gian hoàn thành và chi phí trên một đơn vị hiệu năng.

### 8.5.2. Project Cost Forecasting

Cell 28 tạo forecast chi phí dự án:

| Scenario | Cost |
|---|---:|
| Best case | $2,578.82 |
| Base expected cost | $3,551.20 |
| Contingency 20% | $710.24 |
| Expected with contingency | $4,261.44 |
| 95% confidence range | $2,913.09 - $5,609.79 |
| Worst case | $5,233.82 |

Nhận xét: forecast cho thấy expected cost sau khi cộng contingency là **$4,261.44**. Tuy nhiên, khoảng tin cậy 95% có upper bound **$5,609.79**, cao hơn đáng kể so với expected. Điều này chứng minh tầm quan trọng của contingency và confidence interval trong lập ngân sách GPU: nếu chỉ nhìn vào base expected cost, dự án có thể đánh giá thấp rủi ro vượt ngân sách.

### 8.5.3. Optimization Opportunity Analysis

Cell 29 phân tích các cơ hội tối ưu:

- **Baseline cost:** $1,468.00
- **Total potential savings:** $1,288.32, tương đương **87.8%**
- **Optimized cost:** $179.68

Roadmap được đề xuất:

- **Quick Wins:** Switch to Mixed Precision AMP, Optimize Batch Size.
- **Medium Term:** Implement Early Stopping.
- **Long Term:** Use Spot Instances, Switch to More Efficient GPU Type.

Nhận xét: roadmap này hợp lý vì ưu tiên các kỹ thuật ít rủi ro và dễ triển khai trước. AMP và batch size tuning thường có effort thấp, không phụ thuộc nhiều vào hạ tầng và có thể kiểm chứng nhanh. Các thay đổi như spot instance hoặc đổi GPU type có savings lớn nhưng thường yêu cầu đánh giá thêm về độ ổn định, khả năng checkpoint và tính tương thích.

### 8.5.4. Integrated Cost Dashboard

Cell 30 tạo dashboard nâng cao gồm 6 panel và lưu thành `advanced_finops_dashboard.png`. Dashboard tích hợp nhiều góc nhìn:

1. Multi-GPU cost curve.
2. Scaling efficiency.
3. Project forecast với confidence bands.
4. Phase cost breakdown.
5. Optimization prioritization matrix.
6. Cumulative savings roadmap.

Nhận xét: dashboard tích hợp giúp ra quyết định tốt hơn vì kết hợp cả thông tin kỹ thuật và tài chính. Thay vì xem riêng từng biểu đồ, người quản lý có thể thấy đồng thời tác động của scaling, forecast, phase cost và roadmap tiết kiệm.

### 8.5.5. Challenge Exercise: Cost Optimization Strategy Design

Cell 31 đưa ra kịch bản fine-tuning Large Language Model:

- Baseline: **8x A100** trong **200h**.
- Budget: **$5,000**.
- Deadline: **2 weeks**.
- Baseline cost: **$5,872.00**.

Notebook đề xuất:

- Recommended GPU count for cost efficiency: **16 GPU**.
- Selected strategies:
  1. Switch to Mixed Precision AMP - savings 25%, effort LOW, risk LOW.
  2. Optimize Batch Size - savings 15%, effort LOW, risk LOW.
  3. Implement Early Stopping - savings 20%, effort MEDIUM, risk LOW.
  4. Switch to More Efficient GPU Type - savings 40%, effort HIGH, risk MEDIUM.

Kết quả cost estimate:

| Chỉ số | Giá trị |
|---|---:|
| Direct baseline savings | $4,075.17 (69.4%) |
| Remaining direct cost | $1,796.83 |
| Forecast with phases | $1,527.99 |
| Optimized forecast cost | $467.57 |
| Budget target | $5,000.00 |
| Under budget | YES |
| Risk constraint met | YES |

Nhận xét: chiến lược được chọn tránh các phương án high-risk như phụ thuộc hoàn toàn vào spot preemption, đồng thời ưu tiên các tối ưu có độ tin cậy cao như AMP, batch size tuning và early stopping. Kết quả optimized forecast cost **$467.57** thấp hơn nhiều so với budget **$5,000**, cho thấy việc áp dụng FinOps từ đầu dự án có thể thay đổi đáng kể tính khả thi tài chính của workload LLM.

---

## 3. Kết luận và học hỏi

### 3.1. Những kỹ năng FinOps đã học

Qua bài lab, em đã thực hành các kỹ năng quan trọng trong GPU FinOps:

- Theo dõi GPU cluster bằng các chỉ số như utilization, memory, power, temperature, busy/idle GPU.
- Ghi nhận chi phí workload dựa trên loại GPU, số GPU, thời gian chạy và pricing model.
- Phân biệt workload phù hợp với on-demand và workload phù hợp với spot.
- Đánh giá autoscaling policy dựa trên threshold và trạng thái cluster.
- Tạo cost snapshot, waste report và dashboard để phát hiện lãng phí.
- So sánh hiệu quả tài chính giữa FP32 và Mixed Precision AMP trên GPU thật.
- Forecast chi phí dự án và xây dựng contingency để giảm rủi ro vượt ngân sách.
- Thiết kế roadmap tối ưu chi phí theo mức độ savings, effort và risk.

### 3.2. Các chiến lược cost optimization hiệu quả

Từ kết quả bài lab, các chiến lược tối ưu hiệu quả nhất gồm:

1. **Mixed Precision AMP**  
   Đây là chiến lược có hiệu quả rõ ràng nhất trong phần real GPU training. AMP giúp training nhanh hơn **2.10x**, giảm peak memory **0.22 GB** và giảm chi phí **52.4%** so với FP32.

2. **Sử dụng spot instance cho workload fault-tolerant**  
   Spot instance giúp tiết kiệm đáng kể, ví dụ spot savings report đạt **61.8%**, workflow tổng thể đạt spot savings **70.0%**. Tuy nhiên cần áp dụng cho workload có checkpoint/retry để kiểm soát rủi ro preemption.

3. **Giảm idle GPU và waste**  
   Waste ban đầu đạt **23.2%**, tương ứng potential monthly save **$2,289.51**. Khi workload được tăng và cluster được sử dụng tốt hơn, waste giảm xuống **4.9%**. Điều này cho thấy việc tăng utilization hợp lý và scale đúng thời điểm có tác động lớn đến chi phí.

4. **Autoscaling theo threshold và cost-aware policy**  
   Autoscaler giữ nguyên node khi utilization **49.2%** nằm trong ngưỡng hợp lý, và quyết định scale up khi utilization tăng lên **75.3%** vượt threshold **70.0%**. Cách tiếp cận này giúp cân bằng hiệu năng và chi phí.

5. **Forecast và roadmap optimization**  
   Phần advanced analysis cho thấy chỉ nhìn chi phí hiện tại là chưa đủ. Forecast với confidence interval và roadmap savings giúp lập kế hoạch ngân sách tốt hơn cho các dự án dài hạn.

### 3.3. Ứng dụng thực tế trong projects

Trong dự án AI/ML thực tế, các kỹ thuật từ bài lab có thể được áp dụng như sau:

- Thiết lập dashboard theo dõi GPU utilization, idle cost và workload cost theo thời gian thực.
- Gắn cost attribution theo project, experiment hoặc user để biết workload nào tạo ra chi phí nhiều nhất.
- Áp dụng AMP mặc định cho các mô hình tương thích để giảm thời gian và chi phí training.
- Dùng spot instance cho batch training, hyperparameter tuning hoặc job có checkpoint.
- Thiết kế autoscaling policy dựa trên utilization, queue length và budget guardrail.
- Tạo forecast trước khi chạy các experiment lớn, đặc biệt với LLM fine-tuning hoặc multi-GPU training.
- Đánh giá optimization theo ma trận savings/effort/risk thay vì chỉ chọn phương án tiết kiệm nhiều nhất.

### 3.4. Tổng kết

Bài lab cho thấy GPU FinOps không chỉ là việc giảm chi phí, mà là quá trình liên tục đo lường, phân tích, tối ưu và kiểm chứng lại tác động. Kết quả nổi bật nhất là Mixed Precision AMP giúp giảm **52.4%** chi phí training thật, spot instance có thể tiết kiệm khoảng **60-70%**, và việc giảm waste giúp tiết kiệm đáng kể khi quy mô cluster tăng. Khi kết hợp monitoring, billing, autoscaling, dashboard và forecasting, nhóm kỹ thuật có thể vận hành GPU hiệu quả hơn, kiểm soát ngân sách tốt hơn và vẫn đảm bảo hiệu năng cho workload AI/ML.
