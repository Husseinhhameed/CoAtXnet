def ModifiedDSACLoss(y_true, y_pred, weight_translation_initial=1.0, weight_quaternion_initial=1.0):
    # Split the true and predicted values into delta positions and quaternions
    true_positions, true_quaternions = torch.split(y_true, [3, 4], dim=-1)
    pred_positions, pred_quaternions = torch.split(y_pred, [3, 4], dim=-1)

    # Normalize the predicted quaternions
    pred_quaternions = F.normalize(pred_quaternions, p=2, dim=-1)

    # Calculate the translation error (L2 distance)
    t_error = torch.sqrt(torch.sum((true_positions - pred_positions) ** 2, dim=-1))
    mean_t_error = torch.mean(t_error)

    # Calculate the quaternion angle error
    dot_product = torch.sum(true_quaternions * pred_quaternions, dim=-1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)  # Ensure dot product is within [-1, 1] to avoid NaNs in acos
    angle_error = 2.0 * torch.acos(torch.abs(dot_product))
    mean_q_error = torch.mean(angle_error)

    # Dynamic weight adjustment based on the error ratio
    error_ratio = mean_t_error / (mean_q_error + 1e-8)  # Adding a small epsilon to avoid division by zero
    weight_translation = weight_translation_initial * error_ratio
    weight_quaternion = weight_quaternion_initial / error_ratio

    # Apply dynamic weights to the respective errors
    weighted_t_error = weight_translation * mean_t_error
    weighted_q_error = weight_quaternion * mean_q_error

    # Final combined weighted error
    combined_weighted_error = weighted_t_error + weighted_q_error

    return combined_weighted_error
