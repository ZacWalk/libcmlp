#include "cmlp.h"
#include <stdio.h>

int main(void)
{
    const int layers[] = {2, 3, 1};
    const float input[] = {1.0f, -1.0f};
    float output;
    cmlp_model *model = NULL;
    cmlp_config config = cmlp_default_config();
    cmlp_status status;
    config.layers = layers;
    config.layer_count = 3;
    config.output_activation = CMLP_LINEAR;
    config.seed = 12345;
    status = cmlp_create(&config, &model);
    if (status == CMLP_OK) status = cmlp_forward(model, input, 1, &output);
    cmlp_destroy(model);
    if (status != CMLP_OK) fprintf(stderr, "%s\n", cmlp_status_string(status));
    return status == CMLP_OK ? 0 : 1;
}
