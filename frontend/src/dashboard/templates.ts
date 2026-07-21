export type ProjectTemplate = {
  id: string;
  label: string;
  domain: string;
  description: string;
  sample?: boolean;
};

export const PROJECT_TEMPLATES: ProjectTemplate[] = [
  { id: "blank", label: "Blank project", domain: "Banking", description: "Start from scratch with your own dataset." },
  {
    id: "churn",
    label: "Customer Churn Prediction",
    domain: "Banking",
    description: "Predict which customers are likely to leave.",
    sample: true,
  },
  {
    id: "propensity",
    label: "Purchase Propensity",
    domain: "Retail",
    description: "Score likelihood to purchase a product or service.",
    sample: true,
  },
  {
    id: "coupon",
    label: "Coupon Redemption",
    domain: "Retail",
    description: "Predict coupon uptake and redemption rates.",
    sample: true,
  },
  {
    id: "campaign",
    label: "Campaign Response",
    domain: "Marketing",
    description: "Forecast marketing campaign conversion.",
    sample: true,
  },
  {
    id: "fraud",
    label: "Fraud Detection",
    domain: "Banking",
    description: "Flag suspicious transactions or applications.",
    sample: true,
  },
  {
    id: "credit",
    label: "Credit Approval",
    domain: "Banking",
    description: "Approve or decline credit applications.",
    sample: true,
  },
  {
    id: "loan_default",
    label: "Loan Default",
    domain: "Banking",
    description: "Predict loan default risk on historical portfolios.",
    sample: true,
  },
  {
    id: "cross_sell",
    label: "Cross-sell / Upsell",
    domain: "Retail",
    description: "Identify customers ready for the next best offer.",
    sample: true,
  },
  {
    id: "renewal",
    label: "Membership Renewal",
    domain: "Retail",
    description: "Predict subscription or membership renewals.",
    sample: true,
  },
  {
    id: "returns",
    label: "Return Prediction",
    domain: "Retail",
    description: "Estimate product return probability.",
    sample: true,
  },
  {
    id: "insurance",
    label: "Insurance Claim Prediction",
    domain: "Insurance",
    description: "Model claim likelihood or severity.",
    sample: true,
  },
  {
    id: "hvc",
    label: "High Value Customer",
    domain: "Retail",
    description: "Segment and score high-value customer potential.",
    sample: true,
  },
  {
    id: "store_visit",
    label: "Store Visit Prediction",
    domain: "Retail",
    description: "Predict in-store visit or footfall conversion.",
    sample: true,
  },
  {
    id: "promo",
    label: "Promotion Effectiveness",
    domain: "Marketing",
    description: "Measure promotion lift and response.",
    sample: true,
  },
];

export function templateById(id: string): ProjectTemplate | undefined {
  return PROJECT_TEMPLATES.find((t) => t.id === id);
}
