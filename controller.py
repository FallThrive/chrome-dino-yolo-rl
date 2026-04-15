DEBUG = True


def get_action(detections):
    """
    Determines the action to take based on game detections.
    Could be later replaced with a more sophisticated model, like
    an evolution algorithm or a neural network.

    Args:
        detections: A supervision.Detections object from the inference model.

    Returns:
        A string representing the action: "up" (jump), "down" (duck), or None.
    """
    for i in range(len(detections.xyxy)):
        box = detections.xyxy[i]
        y_centroid = (box[1] + box[3]) / 2
        class_name = detections.data['class_name'][i]

        if DEBUG:
            print(f"[{class_name}] x_left={box[0]:.1f}, y_top={box[1]:.1f}, "
                  f"x_right={box[2]:.1f}, y_bottom={box[3]:.1f}, "
                  f"y_centroid={y_centroid:.1f}")

        if class_name == 'cactus':
            y_in_range = 100 < y_centroid < 190
            x_in_range = 100 < box[0] < 190
            if DEBUG:
                print(f"  -> cactus check: y_in_range={y_in_range} (100<{y_centroid:.1f}<190), "
                      f"x_in_range={x_in_range} (100<{box[0]:.1f}<190)")
            if not y_in_range:
                continue
            if x_in_range:
                if DEBUG:
                    print("  -> ACTION: JUMP!")
                return "up"

        elif class_name == 'bird':
            x_in_range = 100 < box[0] < 190
            if DEBUG:
                print(f"  -> bird check: x_in_range={x_in_range} (100<{box[0]:.1f}<190), "  
                      f"y_centroid={y_centroid:.1f} (threshold=150)")
            if not x_in_range:
                continue
            if y_centroid > 150:
                if DEBUG:
                    print("  -> ACTION: JUMP (low bird)!")
                return "up"
            else:
                if DEBUG:
                    print("  -> ACTION: DUCK (high bird)!")
                return "down"

    return None
