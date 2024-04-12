from train import *
from EUAR import EUAR

if __name__ == '__main__':

    with open(f"./datasets/{args.dataset}.pkl", "rb") as handle:
        data = pickle.load(handle)

    test_data = data["test"]
    test_dataset = get_appropriate_dataset(test_data)
    test_data_loader = DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True,
        )
    multimodal_config = MultimodalConfig(
        beta_shift=args.beta_shift, dropout_prob=args.dropout_prob
    )
    model = EUAR(multimodal_config=multimodal_config, num_labels=1)
    model.load_state_dict(torch.load("./model.pth"))
    model = model.to(DEVICE)
    # model=torch.load("save/model.pth").to(DEVICE)

    test_acc2, test_mae, test_corr, test_f_score, test_acc7 = test_score_model(model, test_data_loader)
    print(
                "best mae:{:.4f}, acc:{:.4f}, acc7:{:.4f}, f1:{:.4f}, corr:{:.4f}".format(
                    test_mae, test_acc2, test_acc7, test_f_score, test_corr
                )
            )