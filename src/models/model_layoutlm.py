# from transformers import LayoutLMForTokenClassification
from transformers.models.layoutlm import LayoutLMForTokenClassification

model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased", num_labels=9)
# data = 

# model.train()
# for epoch in range(1):
#   for batch in tqdm(train_dataloader, desc="Training"):
#       input_ids = batch[0].to(device)
#       bbox = batch[4].to(device)
#       attention_mask = batch[1].to(device)
#       token_type_ids = batch[2].to(device)
#       labels = batch[3].to(device)

#       # forward pass
#       outputs = model(input_ids=input_ids, bbox=bbox, attention_mask=attention_mask, token_type_ids=token_type_ids,
#                       labels=labels)
#       loss = outputs.loss

#       # print loss every 100 steps
#       if global_step % 100 == 0:
#         print(f"Loss after {global_step} steps: {loss.item()}")

#       # backward pass to get the gradients 
#       loss.backward()

#       #print("Gradients on classification head:")
#       #print(model.classifier.weight.grad[6,:].sum())

#       # update
#       optimizer.step()
#       optimizer.zero_grad()
#       global_step += 1