import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy


def compute_confusion_matrix(model, test_loader, device, num_classes):
    model.eval()
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for t, p in zip(labels, predicted):
                cm[t.long(), p.long()] += 1
    return cm


def per_class_metrics(cm, class_names):
    num_classes = cm.shape[0]
    results = {}
    for i in range(num_classes):
        tp = cm[i, i].item()
        fn = cm[i, :].sum().item() - tp
        fp = cm[:, i].sum().item() - tp
        tn = cm.sum().item() - tp - fn - fp

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        results[class_names[i]] = {'TPR': tpr, 'FPR': fpr, 'Precision': precision}
    return results


def evaluate_detailed(model, test_loader, device, class_names):
    num_classes = len(class_names)

    model.eval()
    correct = 0
    total = 0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for t, p in zip(labels, predicted):
                cm[t.long(), p.long()] += 1

    accuracy = 100 * correct / total
    metrics = per_class_metrics(cm, class_names)

    print(f'{"="*60}')
    print(f'  Test Accuracy: {accuracy:.2f}%')
    print(f'{"="*60}')
    print(f'  {"Class":<15} {"TPR(Recall)":>10} {"FPR":>10} {"Precision":>10}')
    print(f'  {"-"*45}')
    for name in class_names:
        m = metrics[name]
        print(f'  {name:<15} {m["TPR"]:>10.4f} {m["FPR"]:>10.4f} {m["Precision"]:>10.4f}')
    print(f'{"="*60}')

    return accuracy, cm, metrics


def get_all_probas_and_labels(model, test_loader, device, num_classes):
    """Return all softmax probabilities (N, C) and true labels (N)."""
    model.eval()
    all_probas = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probas = torch.softmax(outputs, dim=1)
            all_probas.append(probas.cpu())
            all_labels.append(labels)
    return torch.cat(all_probas).numpy(), torch.cat(all_labels).numpy()


def compute_roc_auc_scores(probas, true_labels):
    """Return per-class ROC-AUC scores."""
    n_classes = probas.shape[1]
    scores = {}
    for i in range(n_classes):
        y_true = (true_labels == i).astype(int)
        scores[f'class_{i}'] = roc_auc_score(y_true, probas[:, i])
    scores['macro'] = sum(scores.values()) / n_classes
    return scores


def compute_pr_auc_scores(probas, true_labels):
    """Return per-class PR-AUC (Average Precision) scores."""
    n_classes = probas.shape[1]
    scores = {}
    for i in range(n_classes):
        y_true = (true_labels == i).astype(int)
        scores[f'class_{i}'] = average_precision_score(y_true, probas[:, i])
    scores['macro'] = sum(scores.values()) / n_classes
    return scores


def weak_class_report(cm, class_names, error_threshold=0.08, top_confused=5):
    """Per-class deep dive on classes with error rate > threshold."""
    num_classes = cm.shape[0]
    total_samples = cm.sum().item()
    reports = []

    for i in range(num_classes):
        tp = cm[i, i].item()
        row_total = cm[i, :].sum().item()
        col_total = cm[:, i].sum().item()
        err_rate = 1 - tp / row_total if row_total > 0 else 0

        if err_rate < error_threshold:
            continue

        row_dist = [(cm[i, j].item(), j) for j in range(num_classes) if j != i and cm[i, j] > 0]
        row_dist.sort(reverse=True)
        col_dist = [(cm[j, i].item(), j) for j in range(num_classes) if j != i and cm[j, i] > 0]
        col_dist.sort(reverse=True)

        reports.append((err_rate, i, row_dist, col_dist, row_total, col_total, tp))

    if not reports:
        print(f'  No classes above error threshold {error_threshold:.0%}')
        return

    reports.sort(reverse=True)

    for err_rate, i, row_dist, col_dist, row_total, col_total, tp in reports:
        print(f'\n{"="*65}')
        print(f'  ❖ {class_names[i]} — Error rate: {err_rate:.1%}')
        print(f'{"="*65}')
        print(f'  Support: {row_total} true samples')
        print(f'  TPR: {tp/row_total:.1%}, Precision: {tp/col_total:.1%}' if col_total > 0 else 'N/A')

        print(f'\n  → Predicted AS (row — where true {class_names[i]} goes):')
        print(f'     {"Predicted":<15} {"Count":>6} {"% of class":>10}')
        print(f'     {"-"*33}')
        for c, j in row_dist[:top_confused]:
            print(f'     → {class_names[j]:<15} {c:>6} {100*c/row_total:>9.1f}%')
        missing = row_total - tp - sum(c for c, _ in row_dist[:top_confused])
        if missing > 0:
            print(f'     → (others){"":<7} {missing:>6} {100*missing/row_total:>9.1f}%')

        print(f'\n  → Confused AS (col — false predictions of {class_names[i]}):')
        print(f'     {"True label":<15} {"Count":>6} {"% of pred":>10}')
        print(f'     {"-"*33}')
        for c, j in col_dist[:top_confused]:
            print(f'     ← {class_names[j]:<15} {c:>6} {100*c/col_total:>9.1f}%')
        other_col = col_total - tp - sum(c for c, _ in col_dist[:top_confused])
        if other_col > 0:
            print(f'     ← (others){"":<15} {other_col:>6} {100*other_col/col_total:>9.1f}%')


def analyze_errors(cm, class_names, top_n=5):
    num_classes = cm.shape[0]
    errors = cm.sum().item() - cm.trace().item()

    print(f'{"="*60}')
    print(f'  Error Analysis — Total misclassifications: {errors} / {cm.sum().item()}')
    print(f'  Error rate: {100 * errors / cm.sum().item():.2f}%')
    print(f'{"="*60}')

    # Per-class error count sorted by most errors
    print(f'\n  Per-class errors (most to least):')
    print(f'  {"Class":<15} {"Errors":>8} {"Error %":>8} {"Top confused as":<40}')
    print(f'  {"-"*72}')
    class_errors = []
    for i in range(num_classes):
        tp = cm[i, i].item()
        total = cm[i, :].sum().item()
        err = total - tp
        if err == 0:
            continue
        # Find top confused pairs
        confused = [(cm[i, j].item(), j) for j in range(num_classes) if j != i and cm[i, j] > 0]
        confused.sort(reverse=True)
        top_str = ', '.join([f'{class_names[j]} ({c})' for c, j in confused[:top_n]])
        class_errors.append((err, i, top_str))

    class_errors.sort(reverse=True)
    for err, i, top_str in class_errors:
        total = cm[i, :].sum().item()
        print(f'  {class_names[i]:<15} {err:>8} {100*err/total:>7.1f}%  {top_str:<40}')

    # Most common confusion pairs overall
    print(f'\n  Top confusion pairs (true → predicted):')
    print(f'  {"#":>3} {"True":<15} {"→ Predicted":<15} {"Count":>6}')
    print(f'  {"-"*42}')
    pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j].item(), i, j))
    pairs.sort(reverse=True)
    for rank, (c, i, j) in enumerate(pairs[:top_n * 2], 1):
        print(f'  {rank:>3} {class_names[i]:<15} → {class_names[j]:<15} {c:>6}')

    return class_errors, pairs


def print_comparison_table(acc_mlp, acc_cnn, roc_mlp, roc_cnn, pr_mlp, pr_cnn,
                           mlp_params, cnn_params, class_names):
    print(f'{"="*75}')
    print(f'{"Metric":<25} {"MLP":>12} {"CNN":>12} {"Diff":>12}')
    print(f'{"-"*75}')
    print(f'{"Test Accuracy (%)":<25} {acc_mlp:>11.2f} {acc_cnn:>11.2f} {acc_cnn - acc_mlp:>+11.2f}')
    print(f'{"Macro ROC-AUC":<25} {roc_mlp["macro"]:>11.4f} {roc_cnn["macro"]:>11.4f} {roc_cnn["macro"] - roc_mlp["macro"]:>+11.4f}')
    print(f'{"Macro PR-AUC (AP)":<25} {pr_mlp["macro"]:>11.4f} {pr_cnn["macro"]:>11.4f} {pr_cnn["macro"] - pr_mlp["macro"]:>+11.4f}')
    print(f'{"Parameters":<25} {mlp_params:>11,} {cnn_params:>11,} {cnn_params - mlp_params:>+11,}')
    print(f'{"="*75}')
    print(f'\n{"Per-class ROC-AUC":<25} {"MLP":>12} {"CNN":>12}')
    print(f'{"-"*50}')
    for i, name in enumerate(class_names):
        print(f'{name:<25} {roc_mlp[f"class_{i}"]:>11.4f} {roc_cnn[f"class_{i}"]:>11.4f}')
    print(f'\n{"Per-class PR-AUC (AP)":<25} {"MLP":>12} {"CNN":>12}')
    print(f'{"-"*50}')
    for i, name in enumerate(class_names):
        print(f'{name:<25} {pr_mlp[f"class_{i}"]:>11.4f} {pr_cnn[f"class_{i}"]:>11.4f}')


def ensemble_evaluate(model_a, model_b, test_loader, device, class_names):
    model_a.eval()
    model_b.eval()
    correct = 0
    total = 0
    num_classes = len(class_names)
    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    all_probas = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs_a = model_a(images)
            outputs_b = model_b(images)
            ensemble = torch.softmax(outputs_a, dim=1) + torch.softmax(outputs_b, dim=1)
            _, predicted = torch.max(ensemble, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for t, p in zip(labels, predicted):
                cm[t.long(), p.long()] += 1
            all_probas.append((ensemble / 2).cpu())
            all_labels.append(labels.cpu())

    accuracy = 100 * correct / total
    metrics = per_class_metrics(cm, class_names)
    probas = torch.cat(all_probas).numpy()
    true_labels = torch.cat(all_labels).numpy()

    print(f'{"="*60}')
    print(f'  Ensemble Test Accuracy: {accuracy:.2f}%')
    print(f'{"="*60}')
    print(f'  {"Class":<15} {"TPR(Recall)":>10} {"FPR":>10} {"Precision":>10}')
    print(f'  {"-"*45}')
    for name in class_names:
        m = metrics[name]
        print(f'  {name:<15} {m["TPR"]:>10.4f} {m["FPR"]:>10.4f} {m["Precision"]:>10.4f}')
    print(f'{"="*60}')

    return accuracy, cm, metrics, probas, true_labels
