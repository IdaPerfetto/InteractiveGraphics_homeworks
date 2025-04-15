// Returns a 3x3 transformation matrix as an array of 9 values in column-major order.
// The transformation first applies scale, then rotation, and finally translation.
// The given rotation value is in degrees.

function GetTransform(positionX, positionY, rotation, scale) {

    // We build the three 3x3 transformation matrices as arrays of 9 values in column-major order
    // Let's first create the Scale Matrix
    let ScaleMatrix = Array(9);
    ScaleMatrix[0] = scale; ScaleMatrix[1] = 0;     ScaleMatrix[2] = 0;
    ScaleMatrix[3] = 0;     ScaleMatrix[4] = scale; ScaleMatrix[5] = 0;
    ScaleMatrix[6] = 0;     ScaleMatrix[7] = 0;     ScaleMatrix[8] = 1;

    // We have to convert the rotation value from degrees to radians, then we can define the sine and cosine
    let radians = rotation * Math.PI / 180;
    let cos = Math.cos(radians);
    let sin = Math.sin(radians);

    // Now we create the Rotation Matrix 
    let RotationMatrix = Array(9);
    RotationMatrix[0] = cos;  RotationMatrix[1] = sin; RotationMatrix[2] = 0;
    RotationMatrix[3] = -sin;  RotationMatrix[4] = cos;  RotationMatrix[5] = 0;
    RotationMatrix[6] = 0;    RotationMatrix[7] = 0;    RotationMatrix[8] = 1;

    // Finally we create the Translation Matrix
    let TranslationMatrix = Array(9);
    TranslationMatrix[0] = 1; TranslationMatrix[1] = 0; TranslationMatrix[2] = 0;
    TranslationMatrix[3] = 0; TranslationMatrix[4] = 1; TranslationMatrix[5] = 0;
    TranslationMatrix[6] = positionX; TranslationMatrix[7] = positionY; TranslationMatrix[8] = 1;

    // We now have to apply the transformations in the order: scale -> rotation -> translation
    let Transformation = ApplyTransform(ScaleMatrix, RotationMatrix);
    Transformation = ApplyTransform(Transformation, TranslationMatrix);

    return Transformation;
}

// Returns a 3x3 transformation matrix as an array of 9 values in column-major order.
// The arguments are transformation matrices in the same format.
// The returned transformation first applies trans1 and then trans2.

function ApplyTransform(trans1, trans2) {
    let transform = Array(9);
    for (let row = 0; row < 3; row++) {
        for (let col = 0; col < 3; col++) {
            // Since we are storing the values in column-major order, we calculate the index as col * 3 + row
            // We multiply trans2 * trans1 because trans1 to mantain the rule: rightmost matrix is applied first
            transform[col * 3 + row] =
                trans2[0 * 3 + row] * trans1[col * 3 + 0] +
                trans2[1 * 3 + row] * trans1[col * 3 + 1] +
                trans2[2 * 3 + row] * trans1[col * 3 + 2];
        }
    }

    return transform;
}
